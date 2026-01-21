import torch
import torch.distributed as dist
from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    x = torch.zeros(4, device=device)

    if rank == 0:
        x.fill_(42.0)

    print(f"[rank {rank}] before broadcast: {x}")

    dist.broadcast(x, src=0)
    torch.cuda.synchronize(device)

    print(f"[rank {rank}] after  broadcast: {x}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
