import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo",rank=rank,world_size=world_size)

def distributed_demo(rank,world_size):
    print(f'demo function {rank}')
    setup(rank,world_size)
    data = torch.randint(0,10,(world_size*2,))
    print(f"rank {rank} data (before broadcast): {data}")
    # if dist.get_rank() == 0: # master
    #     data = torch.randint(0,10,,))
    # else:
    #     data = None

    dist.broadcast(tensor=data,src=0,async_op=False)

    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data,async_op=False) # operation sum
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    print(torch.distributed.is_available())
    # print(torch.distributed.is_mpi_available())
    print(torch.distributed.is_gloo_available())
    # dist.destroy_process_group()
    # run 
    world_size = 4 # 4 processes
    mp.spawn(fn=distributed_demo,args=(world_size, ), nprocs=world_size, join=True)
    # dist.destroy_process_group()