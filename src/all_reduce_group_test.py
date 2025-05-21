import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo",rank=rank,world_size=world_size)


def run(rank,world_size):
    print(f'Run function {rank}')
    setup(rank, world_size) # initilize processes
    data = torch.randint(0,10, (5,))
    print(f"rank {rank} data (before reduce all group): {data}")

    # create new group
    sub_group = dist.new_group([0,1]) # group only 0,1 ranks
    dist.all_reduce(data,op=dist.ReduceOp.SUM, group=sub_group)
    print(f"rank {rank} data (after reduce all group): {data}")


if __name__ == "__main__":
    world_size = 4 # 4 processes
    mp.spawn(fn=run,args=(world_size, ), nprocs=world_size, join=True)