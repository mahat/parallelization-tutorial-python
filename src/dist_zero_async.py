import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset

import os

class AsyncDDPWrapper(nn.Module):
    def __init__(self, model, rank, world_size):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self._handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        for param in self.model.parameters():
            if param.requires_grad:
                def hook(param):
                    h = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                    self._handles.append(h)
                param.register_post_accumulate_grad_hook(hook)
                # param.register_hook(hook)
    
    def forward(self,x):
        return self.model(x)

    def sync(self):
        for h in self._handles:
            # wait for sync
            h.wait()
        # average
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data /= self.world_size

        self._handles = []

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


def run(rank,world_size,args):
    setup(rank, world_size) # initilize processes
    torch.manual_seed(42)
    model = SimpleNet().to(device)
    # wrapper
    ddp_model = AsyncDDPWrapper(model,rank,world_size)
    dataset = TensorDataset(X,y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    num_epochs = args['epochs']
    criterion = nn.CrossEntropyLoss()
    ddp_model.train()
    print(f"\n Rank {rank} len train {len(dataloader)}")

    dist.barrier()

    for epoch in range(num_epochs):
         # check eval set beginning of each epoch to validate models are scync
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            #sync 
            ddp_model.sync()
            # Update weights
            optimizer.step()

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
