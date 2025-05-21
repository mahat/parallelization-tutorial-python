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
    # data = torch.randint(0,10,(world_size*2,))
    # tensor_in = torch.arange(world_size * 2, dtype=torch.int64)
    tensor_in = torch.randint(0,10,(world_size*2,))
    print(f"rank {rank} data (before broadcast): {tensor_in}")
    tensor_out = torch.empty(2, dtype=torch.int64)
    # if dist.get_rank() == 0: # master
    #     data = torch.randint(0,10,,))
    # else:
    #     data = None

    print(f"rank {rank} data (before reduce scatter): {tensor_out}")
    dist.reduce_scatter_tensor(tensor_out, tensor_in)
    print(f"rank {rank} data (after reduce scatter): {tensor_out}")
   

if __name__ == "__main__":
    # print(torch.distributed.is_available())
    # print(torch.distributed.is_mpi_available())
    # print(torch.distributed.is_gloo_available())
    # dist.destroy_process_group()
    # run 
    world_size = 4 # 4 processes
    mp.spawn(fn=distributed_demo,args=(world_size, ), nprocs=world_size, join=True)
    # dist.destroy_process_group()