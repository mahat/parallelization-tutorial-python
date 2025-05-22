import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from datetime import datetime

done = mp.Event()

def setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo",rank=rank,world_size=world_size)


def write_to_queue(content,update_queue):
    """Write content to a file in non-blocking way."""
    # with open('./logs/log1.txt', "a") as f:
    #     f.write(str(content) + "\n")
    update_queue.put_nowait({'rank':content['rank'], 'before_reduce':content['before_reduce']})
    # done.wait() # wait for the queue 

def run(rank,world_size,update_queue):
    print(f'Run function {rank} on: {os.getcwd()}')
    setup(rank, world_size) # initilize processes
    data = torch.randint(0,10, (5,))
    print(f"rank {rank} data (before reduce all group): {data}")
    # publish to queue
    write_to_queue({'rank':rank,'time': datetime.now(), 'sum':data.sum(),'before_reduce':True},update_queue)
    # create new group
    # sub_group = dist.new_group([0,1]) # group only 0,1 ranks
    dist.all_reduce(data,op=dist.ReduceOp.SUM)
    write_to_queue({'rank':rank,'time': datetime.now(), 'sum':data.sum(),'before_reduce':False},update_queue)
    print(f"rank {rank} data (after reduce all group): {data}")

if __name__ == "__main__":
    world_size = 4 # 4 processes
    update_queue = mp.Queue()
    mp.spawn(fn=run,args=(world_size, update_queue,), nprocs=world_size, join=True)
    while not update_queue.empty():
        update_result = update_queue.get()
        print(update_result)
    update_queue.close() # close the connection
    done.set()
