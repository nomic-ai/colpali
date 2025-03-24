import torch
import torch.distributed as dist
import torch.distributed.nn


def gather_with_grad(t):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    return torch.cat(torch.distributed.nn.all_gather(t), dim=0)