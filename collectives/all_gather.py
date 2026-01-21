import torch
import torch.distributed as dist

from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # 每个 rank 只“拥有”自己的一小块数据
    x = torch.tensor([rank + 1.0], device=device)

    # 为 all_gather 准备输出容器（每个 rank 一份）
    # 注意此处是正确的 因为all_gather 的 API 是 List[Tensor]
    out = [torch.empty_like(x) for _ in range(world_size)]
    
    # ---- before ----
    print(f"[rank {rank}] before all_gather: x = {x}")

    # ---- all_gather (N -> N, 拼接语义) ----
    dist.all_gather(out, x)

    torch.cuda.synchronize(device)

    # ---- after ----
    print(f"[rank {rank}] after  all_gather: out = {out}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
