import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset

from  torch._utils import _unflatten_dense_tensors, _flatten_dense_tensors

import os

### send if the gradient group is different send async 
## own gradient group self.flat_gradients init
## handle -> receive is async
## after optim step all_gather weights

class AsyncAccumOptimDDPWrapper(nn.Module):
    def __init__(self, model: nn.Module, param_buckets, param_mapping, rank, world_size):
        super().__init__()
        self.model = model
        self.param_buckets = param_buckets
        self.param_mapping = param_mapping
        self.rank = rank
        self.world_size = world_size
        self._handles = []
        self.grad_update_count = {}
        self.optim_grads = torch.zeros(10) # place holder

        self._register_hooks()

    def _register_hooks(self):
        for name, params in self.model.named_parameters():
            if not params.requires_grad:
                continue

            # sync_params = self.param_buckets[self.param_mapping[name]]
            # print(name,self.param_mapping[name],len(sync_params)) 
            def hook(param):
                param_id = id(param)
                update_count = self.grad_update_count.get(param_id,0)
                # check total updates 
                if param_id in self.param_mapping:
                    self.grad_update_count[self.param_mapping[param_id]] = update_count + 1
                else:
                    print('No dispatch')
                    return # no dispatch 
                param_group = self.param_buckets[self.param_mapping[param_id]]
                if self.grad_update_count[self.param_mapping[param_id]] == len(param_group):
                    # flat tensors
                    grads = [p.grad.data for p in param_group]
                    flat_grads = _flatten_dense_tensors(grads)

                    if self.param_mapping[param_id] == dist.get_rank():
                        # calculates its own parameter group and ask other processes to their updates
                        for i in range(self.world_size):
                            recv_buff = torch.zeros_like(flat_grads)
                            if i == dist.get_rank():
                                continue # dont recv from itself
                            sender = dist.recv(recv_buff,src=i)
                            # check recv is all zero
                            if (recv_buff.eq(0).all()) or (recv_buff is None):
                                print(f'RANK: {self.rank} ALL ZEROS recv src={i}')
                            flat_grads += recv_buff.clone()
                        # revert back to original 
                        for uf, p in zip(_unflatten_dense_tensors(flat_grads,grads),param_group):
                            #print(f'RANK: {self.rank} prev grad= {p.grad.data[0:5]}')
                            if p.requires_grad and p.grad is not None:
                                p.grad.data.copy_(uf)
                            else:
                                print('GRAD is not finished')
                    else:
                        # send grads to respective rank
                        if flat_grads.eq(0).all():
                            print(f'RANK: {self.rank} ALL ZEROS sent dst={self.param_mapping[param_id]}')
                        h = dist.isend(flat_grads,dst=self.param_mapping[param_id])
                        self._handles.append(h)
                        # print(f'RANK: {self.rank} sent dst={self.param_mapping[param_id]}')
            params.register_post_accumulate_grad_hook(hook)

    def forward(self,x):
        return self.model(x)

    def sync(self):
        # print('running sync')
        for h in self._handles:
            h.wait()
        # average    
        for param in self.param_buckets[self.rank]:
            if param.requires_grad and param.grad is not None:
                # print(f'RANK: {self.rank} prev grad= {param.grad.data[0:5]}')
                param.grad.data /= self.world_size
                # print(f'RANK: {self.rank} after grad= {param.grad.data[0:5]}')
            else:
                print(param)

        self._handles = []
        self.grad_update_count = {}

    def all_reduce(self):
        ex_params = set([id(p) for p in self.param_buckets[self.rank]])
        for p in self.model.parameters():
            if id(p) not in ex_params:
                # zero data
                p.data.zero_()
        # all reduce
        for p in self.model.parameters():
            dist.all_reduce(p.data,op=dist.ReduceOp.SUM, async_op=False)

    def all_gather(self):
        for r_index,param_group in enumerate(self.param_buckets):
            params_to_send = [p.data for p in param_group]
            # flatten group
            flat_params = _flatten_dense_tensors(params_to_send)
            gathered_params = [torch.zeros_like(flat_params) for _ in range(self.world_size)]
            dist.all_gather(gathered_params, flat_params)
            chosen_param = gathered_params[r_index]
            if chosen_param.eq(0).all():
                print(f'RANK: {self.rank} chosen is ALL ZEROS')
            for uf, p in zip(_unflatten_dense_tensors(chosen_param,params_to_send),param_group):
                p.data.copy_(uf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
# Generate synthetic data for a simple classification task
N = 800
D = 5  # Input dimension
K = 3  # Number of classes

X = torch.randn(N, D)  # Input data
y = torch.randint(0, K, (N,))  # Labels

X_eval = torch.randn(200, D)  # Input data
y_eval = torch.randint(0, K, (200,))  # Labels

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(D, 64)   # First layer (to be trained)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, K)   # Second layer (to be trained)
        # self.fc3 = nn.Linear(64, 64)  # Third layer (not trained)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo",rank=rank,world_size=world_size)

def generate_buckets(model, world_size):
    # buckets layers based on size of the parameters 
    # calculating param loads -> info
    params_info = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_info.append((name, param, len(param)))
    # sort based on size of params
    params_info.sort(key=lambda x: x[2],reverse=True)
    
    # params buckets
    param_buckets = [(0,list()) for i in range(world_size)] # each rank are asigned to list of parameters
    # param mapping 
    param_mapping = {}
    for pg in params_info:
        min_size_in = 0
        for i in range(1,world_size):
            if param_buckets[min_size_in][0] > param_buckets[i][0]:
                min_size_in = i
        # update 
        name, param, size = pg
        param_buckets[min_size_in] = (param_buckets[min_size_in][0] + size, param_buckets[min_size_in][1] + [param]) 
        # param_mapping[name] = min_size_in
        param_mapping[id(param)] = min_size_in
    # remove param size from param_buckets -> cleaner 
    param_buckets = [e[1] for e in param_buckets]
    # print info
    print(param_mapping)
    # print(param_buckets)
    return param_buckets, param_mapping


def run(rank,world_size,args):
    setup(rank, world_size) # initilize processes
    torch.manual_seed(42)
    # calculate buckets
    # bucket_index -> parameters
    model = SimpleNet().to(device)
    # param_name -> bucket_index
    param_buckets, param_mapping = generate_buckets(model, world_size) 
    # wrapper
    ddp_model = AsyncAccumOptimDDPWrapper(model, param_buckets, param_mapping, rank, world_size)
    dataset = TensorDataset(X,y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler)

    # optimizer = optim.AdamW(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(param_buckets[rank], lr=0.001)
    num_epochs = args['epochs']
    criterion = nn.CrossEntropyLoss()
    ddp_model.train()
    print(f"\n Rank {rank} len train {len(dataloader)}")
    dist.barrier()

    for epoch in range(num_epochs):
        ddp_model.train()
         # check eval set beginning of each epoch to validate models are scync
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Backward and optimize
            optimizer.zero_grad()
            # manually zero all other gradients to prevent gradient accumulation
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.data.zero_()

            loss.backward()
            #sync 
            ddp_model.sync()
            # Update weights
            optimizer.step()

            # all_gather weights
            # ddp_model.all_reduce()
            ddp_model.all_gather()

        ddp_model.eval()
        eval_loss = 0.0
        test_loader = DataLoader(TensorDataset(X_eval,y_eval),batch_size=args['batch_size'],shuffle=False)
        with torch.no_grad():  # Disable gradient calculation during evaluation
            for inputs, labels in test_loader:
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * inputs.shape[0]

        print(f"\n Rank {rank} Epoch {epoch + 1}, Epoch Loss: {eval_loss / len(test_loader):.4f}")

if __name__ == "__main__":
    world_size = 4 # 4 processes
    args = {}
    args['epochs'] = 2
    args['batch_size'] = 13

    mp.spawn(fn=run,args=(world_size,args, ), nprocs=world_size, join=True)
