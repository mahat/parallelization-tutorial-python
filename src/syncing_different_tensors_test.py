import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import os

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def run(rank, world_size):
    print(f"Run function {rank} on: {os.getcwd()}")
    setup(rank, world_size)  # initilize processes
    data_1 = torch.randn(size=(5,))
    data_2 = torch.randn(size=(10,))
    
    print(f"rank {rank} data (before reduce all group): {data_1}")
    print(f"rank {rank} data (before reduce all group): {data_2}")

    # average tensors
    dist.all_reduce(data_1, op=dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(data_2, op=dist.ReduceOp.SUM, async_op=False)

    print(f"rank {rank} data (before reduce all group): {data_1}")
    print(f"rank {rank} data (before reduce all group): {data_2}")

if __name__ == "__main__":
    world_size = 4  # 4 processes
    processes = []
    mp.spawn(
        fn=run,
        args=(
            world_size,
        ),
        nprocs=world_size,
        join=True,
    )
    # for rank in range(world_size):
    #     p = Process(target=run, args=(rank, world_size, error_queue))
    #     p.start()
    #     processes.append(p)
    # done.set()  
