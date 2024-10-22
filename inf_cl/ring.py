import os
import math
import random

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
import numpy as np

import triton
import triton.language as tl

from .flash import _flash_prob_forward, _flash_prob_backward, _cal_flash_loss


class RingComm:

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        # print(f'rank: {self.rank}, send_rank: {self.send_rank}, recv_rank: {self.recv_rank}')
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send, recv_tensor = None):
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


class GradientGather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, dx):
        dist.all_reduce(dx)
        return dx


class RingProb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, group):
        rank = dist.get_rank()
        k = k.contiguous()
        comm = RingComm(group)

        colle = [q, k]

        lse = None
        next_k = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k: torch.Tensor = comm.send_recv(k)
                comm.commit()

            # vanilla lse
            qk = torch.einsum("mhd,nhd->mn", q, k)
            block_lse = torch.log(torch.exp(qk).sum(dim=-1))

            if step == 0:
                lse = block_lse
            else:
                lse = lse - F.logsigmoid(lse - block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k

        # this should be out_padded
        colle.append(lse)
        ctx.save_for_backward(*colle)
        ctx.group = group
        return lse

    @staticmethod
    def backward(ctx, dlse):
        rank = dist.get_rank()
        q, k, lse = ctx.saved_tensors
        k_comm = RingComm(ctx.group)
        d_k_comm = RingComm(ctx.group)
        dq, dk = None, None
        next_dk = None

        block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=k.device)

        next_dk, next_k = None, None

        for step in range(k_comm.world_size):
            if step + 1 != k_comm.world_size:
                next_k = k_comm.send_recv(k)
                k_comm.commit()

            # vanilla gradient calculation
            qk = torch.einsum("mhd,nhd->mn", q, k)
            qk_grad = torch.exp(qk - lse[:, None]).float()
            qk_grad = qk_grad * dlse[:, None]
            block_dq_buffer = torch.einsum("mn,nhd->mhd", qk_grad, k.float())
            block_dk_buffer = torch.einsum("nm,mhd->nhd", qk_grad.T, q.float())

            if step == 0:
                dq = block_dq_buffer
                dk = block_dk_buffer
            else:
                dq += block_dq_buffer
                d_k_comm.wait()
                dk = block_dk_buffer + next_dk

            if step + 1 != k_comm.world_size:
                k_comm.wait()
                k = next_k

            next_dk = d_k_comm.send_recv(dk)
            d_k_comm.commit()

        d_k_comm.wait()

        return dq, next_dk, None
    

class InfProb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, group):
        rank = dist.get_rank()
        k = k.contiguous()
        comm = RingComm(group)

        colle = [q, k]

        lse = None
        next_k = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k: torch.Tensor = comm.send_recv(k)
                comm.commit()

            # flash lse
            block_lse = _flash_prob_forward(q, k)

            if step == 0:
                lse = block_lse
            else:
                lse = lse - F.logsigmoid(lse - block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k

        # this should be out_padded
        colle.append(lse)
        ctx.save_for_backward(*colle)
        ctx.group = group
        return lse

    @staticmethod
    def backward(ctx, dlse):
        rank = dist.get_rank()
        q, k, lse = ctx.saved_tensors
        k_comm = RingComm(ctx.group)
        d_k_comm = RingComm(ctx.group)
        dq, dk = None, None
        next_dk = None

        block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=k.device)

        next_dk, next_k = None, None

        for step in range(k_comm.world_size):
            if step + 1 != k_comm.world_size:
                next_k = k_comm.send_recv(k)
                k_comm.commit()

            # flash gradient calculation
            block_dq_buffer, block_dk_buffer = _flash_prob_backward(q, k, lse, dlse)

            if step == 0:
                dq = block_dq_buffer
                dk = block_dk_buffer
            else:
                dq += block_dq_buffer
                d_k_comm.wait()
                dk = block_dk_buffer + next_dk

            if step + 1 != k_comm.world_size:
                k_comm.wait()
                k = next_k

            next_dk = d_k_comm.send_recv(dk)
            d_k_comm.commit()

        d_k_comm.wait()

        return dq, next_dk, None


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)             
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 


def _cal_ring_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    lse = RingProb.apply(q, k, None)
    numerator = torch.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


def _cal_inf_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    lse = InfProb.apply(q, k, None)
    numerator = torch.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


def cal_ring_loss(q, k, labels=None, scale=None, head_dim=256):
    """The triton implementation of the ring-cl.

    Args:
        q (torch.Tensor): The column tensor in contrastive loss. The shape is [B, D].
        k (torch.Tensor): The row tensor in contrastive loss. The shape is [B, D].
        labels (torch.Tensor, optional): In CLIP loss, the labels are the indices of the positive pairs. The shape is [B]. When setting to None, the labels are the range of [0, B). Defaults to None.
        scale (torch.Tensor, optional): The scale tensor of the query tensor. Defaults to None.
        head_dim (int, optional): The head dimension. (must be 16, 32, 64, 128 or 256). Defaults to 256.

    """

    if labels is None:
        labels = torch.arange(q.shape[0]).to(q.device)
    if scale is None:
        scale = 1.0
    else:
        scale = GradientGather.apply(scale)
    if torch.distributed.is_initialized():
        return _cal_ring_loss(scale * q, k, labels, head_dim).mean()
    else:
        return _cal_flash_loss(scale * q, k, labels, head_dim).mean()


def cal_inf_loss(q, k, labels=None, scale=None, head_dim=256):
    """The triton implementation of the inf-cl.

    Args:
        q (torch.Tensor): The column tensor in contrastive loss. The shape is [B, D].
        k (torch.Tensor): The row tensor in contrastive loss. The shape is [B, D].
        labels (torch.Tensor, optional): In CLIP loss, the labels are the indices of the positive pairs. The shape is [B]. When setting to None, the labels are the range of [0, B). Defaults to None.
        scale (torch.Tensor, optional): The scale tensor of the query tensor. Defaults to None.
        head_dim (int, optional): The head dimension. (must be 16, 32, 64, 128 or 256). Defaults to 256.

    """

    if labels is None:
        labels = torch.arange(q.shape[0]).to(q.device)
    if scale is None:
        scale = 1.0
    else:
        scale = GradientGather.apply(scale)
    if torch.distributed.is_initialized():
        return _cal_inf_loss(scale * q, k, labels, head_dim).mean()
    else:
        return _cal_flash_loss(scale * q, k, labels, head_dim).mean()


if __name__ == "__main__":
    import time

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(f'cuda:{os.environ["LOCAL_RANK"]}')

    # Parameters
    dtype = torch.float32
    num_heads = 3        # Number of attention heads
    seq_length_q = 32768 # Sequence length
    seq_length_k = 32768
    d_model = 256        # Dimension of each head (must be 16, 32, 64, or 128)

    # Randomly initialize inputs
    q = torch.rand((seq_length_q // world_size, num_heads * d_model), dtype=dtype, device=f"cuda")
    k = torch.rand((seq_length_k // world_size, num_heads * d_model), dtype=dtype, device=f"cuda")
    l = torch.ones([], dtype=dtype, device="cuda") * np.log(1 / 0.07); l = l.requires_grad_() # Logit scale

    q = F.normalize(q, p=2, dim=-1).requires_grad_() # Query
    k = F.normalize(k, p=2, dim=-1).requires_grad_() # Key

    q1 = q.clone().detach().requires_grad_()
    k1 = k.clone().detach().requires_grad_()
    l1 = l.clone().detach().requires_grad_()

    for i in range(1000):
        # A. local torch gradient
        start = time.time()
        # A.1. gather q, k
        gathered_q = [torch.zeros_like(q) for _ in range(world_size)]
        gathered_k = [torch.zeros_like(k) for _ in range(world_size)]
        dist.all_gather(gathered_q, q)
        dist.all_gather(gathered_k, k)
        gathered_q[rank] = q
        gathered_k[rank] = k
        all_q = torch.cat(gathered_q, dim=0)
        all_k = torch.cat(gathered_k, dim=0)
        # A.2. calculating qk logits
        qk = torch.einsum("md,nd->mn", l.exp() * all_q, all_k)
        kq = qk.T
        _labels = torch.arange(seq_length_q).to(q.device)
        # A.3. calculating loss
        loss_i2t = F.cross_entropy(qk, _labels, reduction="mean")
        loss_t2i = F.cross_entropy(kq, _labels, reduction="mean")
        # A.4. scaling loss to normal value
        scale_factor = (all_q.shape[0] / q.shape[0])
        loss = (loss_i2t + loss_t2i) * 0.5 * scale_factor
        loss.backward()
        show_loss = loss.detach().clone()
        dist.all_reduce(show_loss)
        show_loss = show_loss / (world_size * scale_factor)
        end = time.time()

        dist.barrier()

        # B. triton implementation
        start1 = time.time()
        # labels = torch.arange(seq_length_q // world_size).to(q.device)
        loss1_i2t = cal_inf_loss(q1, k1, scale=l1.exp())
        loss1_t2i = cal_inf_loss(k1, q1, scale=l1.exp())
        loss1 = (loss1_i2t + loss1_t2i).mean() * 0.5
        loss1.backward()
        end1 = time.time()

        dist.barrier()

        if rank == 0:
            print(rank, end - start, end1 - start1, loss, show_loss, loss1)
            print(l.grad, l1.grad, torch.max(torch.abs(q.grad - q1.grad)), torch.max(torch.abs(k.grad - k1.grad)))

        q.grad = None; k.grad = None; l.grad = None
        q1.grad = None; k1.grad = None; l1.grad = None
