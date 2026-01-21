import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_process_group(rank: int, world_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL collectives.")
    if world_size < 2:
        raise RuntimeError("world_size must be >= 2.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def cleanup_process_group():
    dist.destroy_process_group()


def spawn(fn, world_size: int):
    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
