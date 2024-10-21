import math

import torch
import torch.nn.functional as F
import numpy as np

import triton
import triton.language as tl


@triton.jit
def _prob_fwd_kernel(
    Q,
    K,
    LSE,
    nheads,
    seqlen_q,
    seqlen_k,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # start index of sequence length
    start_m = tl.program_id(0)

    # initialize offsets
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    # initialize pointer to m and l
    lse_i    = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i      = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # loop over k, v and update accumulator
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute qk ----
            qk += tl.dot(q, tl.trans(k))

        # Trying to combine the two masks seem to make the result wrong
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        # Fix out of bound access
        p = tl.where((start_n + offs_n)[None, :] < seqlen_k, p, 0.0)
        # -- update statistics
        lse_i = tl.exp(m_i - m_ij) * lse_i + tl.sum(p, 1)
        m_i = m_ij

    lse_i = m_i + tl.log(lse_i)
    # mask out the padded values
    lse_i = tl.where(offs_m < seqlen_q, lse_i, 0.0)

    tl.store(LSE + offs_m, lse_i)


@triton.jit
def _dq_prob_bwd_kernel(
    Q,
    K,
    dQ,
    LSE,
    dLSE,
    nheads,
    seqlen_q,
    seqlen_k,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    # start index of sequence length
    start_m = tl.program_id(0)

    # initialize offsets
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs  = Q  + ndims * offs_m[:, None]
    dq_ptrs = dQ + ndims * offs_m[:, None]
    k_ptrs  = K  + ndims * offs_n[:, None]
    # setting lse
    lse = tl.load(LSE + offs_m, mask=offs_m < seqlen_q, other=0.0)
    dlse = tl.load(dLSE + offs_m, mask=offs_m < seqlen_q, other=0.0)

    # loop over k, v and update accumulator
    end_n = seqlen_k        
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute qk ----
            qk += tl.dot(q, tl.trans(k))

        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk_grad, 0.0)
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, "=r, r", [qk_grad], dtype=tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute q grad ----
            # NOTE: tl.float32 adopt tf32, which causes precision inconsistency with torch
            # A solution for this problem
            # Refer to issue: https://github.com/triton-lang/triton/issues/4574
            # if allow_tf32:
            k = tl.inline_asm_elementwise(ASM, "=r, r", [k], dtype=tl.float32, is_pure=True, pack=1)
            q_grad = tl.dot(qk_grad, k)
            # Another solution for this problem
            # Refer to https://github.com/triton-lang/triton/issues/376
            # q_grad = tl.dot(qk_grad, k.to(tl.float32), allow_tf32=False)
            # -- store dq ----
            dq_h = tl.load(dq_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            tl.store(dq_ptrs + offs_hd, dq_h + q_grad, mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _dk_prob_bwd_kernel(
    Q,
    K,
    dK,
    LSE,
    dLSE,
    nheads,
    seqlen_q,
    seqlen_k,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    # start index of sequence length
    start_n = tl.program_id(0)

    # initialize offsets
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs  = Q  + ndims * offs_m[:, None]
    k_ptrs  = K  + ndims * offs_n[:, None]
    dk_ptrs = dK + ndims * offs_n[:, None]

    # loop over q and update accumulator
    end_m = seqlen_q        
    for start_m in range(0, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        # setting lse
        lse = tl.load(LSE + offs_m + start_m, mask=offs_m < seqlen_q, other=0.0)
        dlse = tl.load(dLSE + offs_m + start_m, mask=offs_m < seqlen_q, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(offs_m + start_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=(offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute qk ----
            qk += tl.dot(q, tl.trans(k))

        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_m + offs_m)[:, None] < seqlen_q, qk_grad, 0.0)
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, "=r, r", [qk_grad], dtype=tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(start_m + offs_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=(offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute k grad ----
            q = tl.inline_asm_elementwise(ASM, "=r, r", [q], dtype=tl.float32, is_pure=True, pack=1)
            k_grad = tl.dot(tl.trans(qk_grad), q)
            # k_grad = tl.dot(tl.trans(qk_grad), q.to(tl.float32))
            # -- store dk ----
            dk_h = tl.load(dk_ptrs + offs_hd, mask=(offs_n)[:, None] < seqlen_k, other=0.0)
            tl.store(dk_ptrs + offs_hd, dk_h + k_grad, mask=(offs_n)[:, None] < seqlen_k)


def _flash_prob_forward(q, k):
    # shape constraints
    seqlen_q, nheads, d = q.shape
    seqlen_k, _, _ = k.shape
    assert k.shape == (seqlen_k, nheads, d)
    # assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype, "All tensors must have the same type"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((seqlen_q_rounded), device=q.device, dtype=torch.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8
    num_stages = 1
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), 1)
    _prob_fwd_kernel[grid](
        q,
        k,
        lse,
        nheads,
        seqlen_q,
        seqlen_k,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    lse = lse[:seqlen_q]

    return lse


def _flash_prob_backward(q, k, lse, dlse):
    # shape constraints
    seqlen_q, nheads, d = q.shape
    seqlen_k, _, _ = k.shape
    assert k.shape == (seqlen_k, nheads, d)
    # assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype, "All tensors must have the same type"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)

    q = q.contiguous()
    k = k.contiguous()
    dlse = dlse.contiguous()

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8
    num_stages = 1
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), 1)
    _dq_prob_bwd_kernel[grid](
        q,
        k,
        dq,
        lse,
        dlse,
        nheads,
        seqlen_q,
        seqlen_k,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    BLOCK_N = BLOCK_M
    BLOCK_M = BLOCK_N
    grid = lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]), 1)
    _dk_prob_bwd_kernel[grid](
        q,
        k,
        dk,
        lse,
        dlse,
        nheads,
        seqlen_q,
        seqlen_k,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dq = dq[:seqlen_q]
    dk = dk[:seqlen_k]

    return dq, dk


class FlashProb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k):
        lse = _flash_prob_forward(q, k)
        ctx.save_for_backward(q, k, lse)

        return lse

    @staticmethod
    def backward(ctx, dlse):
        q, k, lse = ctx.saved_tensors
        dq, dk = _flash_prob_backward(q, k, lse, dlse)

        return dq, dk


def _cal_flash_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    # NOTE: logits forward or backward should keep fp32 for better precision
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    lse = FlashProb.apply(q, k)
    numerator = torch.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


if __name__ == '__main__':
    # Parameters
    num_heads = 2   # Number of attention heads
    seq_length_q = 32768 # Sequence length
    seq_length_k = 32768
    d_model = 256    # Dimension of each head (must be 16, 32, 64, or 128)

    import time

    # Randomly initialize inputs
    q = torch.rand((seq_length_q, num_heads * d_model), dtype=torch.float32, device="cuda") # Query
    k = torch.rand((seq_length_k, num_heads * d_model), dtype=torch.float32, device="cuda") # Key
    l = torch.ones([], device="cuda") * np.log(1 / 0.02)

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    q.requires_grad = True
    k.requires_grad = True
    l.requires_grad = True
    
    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    l1 = l.clone().detach().requires_grad_(True)

    labels = torch.arange(seq_length_q).cuda()

    for i in range(1000):

        # A. torch gradient
        start = time.time()
        qk = torch.einsum("md,nd->mn", l.exp() * q, k)
        loss = F.cross_entropy(qk, labels, reduction="mean")
        loss.backward()
        end = time.time()

        # qk_grad = torch.exp(qk - lse[:, None])
        # qk_grad[torch.arange(len(labels)), labels] -= 1

        # q_grad = torch.einsum("mn,nhd->mhd", qk_grad, _k)
        # k_grad = torch.einsum("nm,mhd->nhd", qk_grad.T, _q)
        # l_grad = q_grad.sum()
        # # print(qk_grad)

        # print("========= Torch Gradient =========")
        # print(end - start, loss)

        # logit_scale_grad = torch.einsum("mn,mhd->qk_grad
        # print(torch.sum(nq.grad * nq))
        # print(nq.grad[:2, 0, :2])
        # print(q_grad, _q.grad)
        # print(logit_scale.grad)
        # print(lse[:2])
        # print(logit_scale.grad)
        # print(_q.grad[:5, :, :5], _k.grad[:5, :, :5])

        # C. triton gradient
        start1 = time.time()
        loss1 = _cal_flash_loss(l1.exp() * q1, k1, labels)
        loss1 = loss1.mean()
        loss1.backward()
        end1 = time.time()

        # print("========= Triton Gradient =========")
        # print(end - start, loss)
        # print(lse[:2])
        # print(lq.grad[:2, 0, :2] * logit_scale.exp(), q.grad[:2, 0, :2])
        # exit(0)
        # print(logit_scale.grad, torch.exp(torch.log(nq.grad) + torch.log(lq)).sum())
        # print(q.grad[:5, :, :5], k.grad[:5, :, :5])

        print("========= Difference =========")
        # print(torch_time, triton_time, torch.max(torch.abs(q.grad - q_grad)), torch.max(torch.abs(k.grad - k_grad)))
        print(end - start, end1 - start1, l.grad, l1.grad, torch.max(torch.abs(q.grad - q1.grad)), torch.max(torch.abs(k.grad - k1.grad)))

        q.grad = None
        k.grad = None
        l.grad = None
        q1.grad = None
        k1.grad = None
        l1.grad = None

        # exit(0)
