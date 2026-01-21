import torch
import torch.distributed as dist

from utils import setup_process_group, cleanup_process_group, spawn


def run(rank: int, world_size: int):
    setup_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    x = torch.tensor([rank + 1.0], device=device)

    # 输出 tensor：注意 size = world_size * x.numel()
    out = torch.empty(world_size * x.numel(), device=device)

    print(f"[rank {rank}] before all_gather_into_tensor: x = {x}")

    # -------- 通信 + 写入 --------
    dist.all_gather_into_tensor(out, x)
    torch.cuda.synchronize(device)

    print(f"[rank {rank}] raw out = {out}")

    # reshape 成“按 rank 堆叠”的形态
    out_view = out.view(world_size, *x.shape)

    print(f"[rank {rank}] reshaped out = {out_view}")

    cleanup_process_group()


if __name__ == "__main__":
    spawn(run, world_size=2)
