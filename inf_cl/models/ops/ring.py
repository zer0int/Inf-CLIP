import math
import random

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
import numpy as np

import triton
import triton.language as tl

from .flash import _flash_prob_forward, _flash_prob_backward


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
    

class RingFlashProb(torch.autograd.Function):

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


def _cal_ring_flash_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    lse = RingFlashProb.apply(q, k, None)
    numerator = torch.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


if __name__ == "__main__":
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # device = torch.device(f"cuda:{rank}")

    set_seed(rank)
    torch.cuda.set_device(rank)

    print(rank, world_size)
    import time

    # Parameters
    dtype = torch.float32
    num_heads = 3   # Number of attention heads
    seq_length_q = 32768 # Sequence length
    seq_length_k = 32768
    d_model = 256    # Dimension of each head (must be 16, 32, 64, or 128)

    l = torch.ones([], dtype=dtype, device="cuda") * np.log(1 / 0.07)

    # Randomly initialize inputs
    q = torch.rand((seq_length_q, num_heads * d_model), dtype=dtype, device=f"cuda") # Query
    k = torch.rand((seq_length_k, num_heads * d_model), dtype=dtype, device=f"cuda") # Key

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    q.requires_grad = True
    k.requires_grad = True
    l.requires_grad = True

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(l, src=0)

    q_chunk_size  = math.ceil(seq_length_q / world_size)
    k_chunk_size  = math.ceil(seq_length_k / world_size)

    q = q.view(seq_length_q, num_heads, d_model).contiguous()
    k = k.view(seq_length_k, num_heads, d_model).contiguous()

    labels = torch.arange(seq_length_q).to(q.device)

    local_q = q.chunk(world_size, dim=0)[rank].detach().clone()
    local_k = k.chunk(world_size, dim=0)[rank].detach().clone()
    local_l = l.detach().clone()

    local_q.requires_grad = True
    local_k.requires_grad = True
    local_l.requires_grad = True

    local_q1 = q.chunk(world_size, dim=0)[rank].detach().clone()
    local_k1 = k.chunk(world_size, dim=0)[rank].detach().clone()
    local_l1 = l.detach().clone()

    local_q1.requires_grad = True
    local_k1.requires_grad = True
    local_l1.requires_grad = True

    local_labels = torch.arange(rank * q_chunk_size, (rank + 1) * q_chunk_size).to(q.device)
    local_labels = local_labels[local_labels < seq_length_q]
    # valid 
    # local_labels = local_labels[local_labels > rank * k_chunk_size]
    local_labels = local_labels - rank * k_chunk_size
    local_labels = local_labels[local_labels >= 0]

    for i in range(1000):
        # B. local torch gradient
        start = time.time()
        gathered_q = [torch.zeros_like(local_q) for _ in range(world_size)]
        gathered_k = [torch.zeros_like(local_k) for _ in range(world_size)]
        dist.all_gather(gathered_q, local_q)
        dist.all_gather(gathered_k, local_k)
        gathered_q[rank] = local_q
        gathered_k[rank] = local_k
        all_q = torch.cat(gathered_q, dim=0)
        all_k = torch.cat(gathered_k, dim=0)
        # 1. calculating qk logits
        qk = torch.einsum("mhd,nhd->mn", local_l.exp() * all_q, all_k)
        kq = qk.T
        loss_i2t = F.cross_entropy(qk, labels, reduction="mean")
        loss_t2i = F.cross_entropy(kq, labels, reduction="mean")
        scale_factor = (all_q.shape[0] / local_q.shape[0])
        loss = (loss_i2t + loss_t2i) * 0.5 * scale_factor
        loss.backward()
        show_loss = loss.detach().clone()
        dist.all_reduce(show_loss)
        show_loss = show_loss / (world_size * scale_factor)
        end = time.time()

        dist.barrier()
        # B. triton
        start1 = time.time()
        local_l1g = GradientGather.apply(local_l1)
        loss1_i2t = _cal_ring_loss(local_l1g.exp() * local_q1, local_k1, local_labels, head_dim=256)
        loss1_t2i = _cal_ring_loss(local_l1g.exp() * local_k1, local_q1, local_labels, head_dim=256)
        loss1 = (loss1_i2t + loss1_t2i).mean() * 0.5
        loss1.backward()
        end1 = time.time()

        if rank == 0:
            print(loss, show_loss, loss1)
            print(rank, end - start, end1 - start1, local_l.grad, local_l1.grad, torch.max(torch.abs(local_q.grad - local_q1.grad)), torch.max(torch.abs(local_k.grad - local_k1.grad)))

        dist.barrier()

        local_q.grad = None
        local_k.grad = None
        local_l.grad = None
        local_q1.grad = None
        local_k1.grad = None
        local_l1.grad = None
