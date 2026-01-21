import torch
import torch.distributed as dist

from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # 每个 rank 一个不同的值，方便看聚合效果
    x = torch.tensor([rank + 1.0], device=device)

    # ---- before ----
    print(f"[rank {rank}] before reduce: {x}")

    # ---- reduce (N -> 1) ----
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)

    # 同步，确保打印的是最终结果
    torch.cuda.synchronize(device)

    # ---- after ----
    print(f"[rank {rank}] after  reduce: {x}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
