import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from inf_cl import cal_inf_loss


def create_cl_tensors(rank, world_size):
    # Parameters
    dtype = torch.float32
    num_heads = 3        # Number of attention heads
    seq_length_q = 32768 # Sequence length
    seq_length_k = 32768
    d_model = 256        # Dimension of each head (must be 16, 32, 64, or 128)

    # Randomly initialize inputs
    q = torch.rand((seq_length_q // world_size, num_heads * d_model), dtype=dtype, device=f"cuda:{rank}")
    k = torch.rand((seq_length_k // world_size, num_heads * d_model), dtype=dtype, device=f"cuda:{rank}")
    l = torch.ones([], dtype=dtype, device=f"cuda:{rank}") * np.log(1 / 0.07)

    q = F.normalize(q, p=2, dim=-1).requires_grad_() # Query
    k = F.normalize(k, p=2, dim=-1).requires_grad_() # Key
    l = l.requires_grad_() # Logit scale

    return q, k, l


if __name__ == "__main__":
    # Assume that the distributed environment has been initialized
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)

    # Exampled by Image-Text Contrastive Learning, q is the global image features, 
    # k is the text features, and l is the logit scale.
    q, k, l = create_cl_tensors(rank, world_size)

    # labels are diagonal elements by default. 
    # labels = torch.arange(q.shape[0])
    loss = cal_inf_loss(q, k, scale=l.exp())

    print(loss)
