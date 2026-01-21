import torch
import torch.distributed as dist

from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 每个 rank 一块数据
    x = torch.tensor([rank + 1.0], device=device)

    # -------- 通信层：只负责收齐 --------
    gathered = [torch.empty_like(x) for _ in range(world_size)]

    print(f"[rank {rank}] before all_gather: x = {x}")

    dist.all_gather(gathered, x)
    torch.cuda.synchronize(device)

    print(f"[rank {rank}] gathered list = {gathered}")

    # -------- 计算层：显式拼接 --------
    y = torch.cat(gathered, dim=0)

    print(f"[rank {rank}] after cat: y = {y}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
