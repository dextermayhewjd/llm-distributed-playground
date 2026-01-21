import torch
import torch.distributed as dist

from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # 每个 rank 一个不同的值，方便验证
    x = torch.tensor([rank + 1.0], device=device)

    # ---- before ----
    print(f"[rank {rank}] before all_reduce: {x}")

    # ---- all_reduce (N -> N) ----
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # 确保通信完成后再打印
    torch.cuda.synchronize(device)

    # ---- after ----
    print(f"[rank {rank}] after  all_reduce: {x}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
