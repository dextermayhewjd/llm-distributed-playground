import torch
import torch.distributed as dist

from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # ---- 每个 rank 一个“不同结构”的全量输入 ----
    # rank 0: [1, 2]
    # rank 1: [3, 4]
    if rank == 0:
        x = torch.tensor([1.0, 2.0], device=device)
    else:
        x = torch.tensor([3.0, 4.0], device=device)

    out = torch.empty(1, device=device)

    print(f"[rank {rank}] before reduce_scatter: x = {x}")

    dist.reduce_scatter(
        out,
        list(x.chunk(world_size)),
        op=dist.ReduceOp.SUM,
    )

    torch.cuda.synchronize(device)

    print(f"[rank {rank}] after  reduce_scatter: out = {out}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
