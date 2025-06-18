import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    
class Partions:
    def __init__(self,X,y,world_size,X_eval,y_eval,fracs=None,):
        self.X = X
        self.y = y

        self.X_eval = X_eval
        self.y_eval = y_eval
        # set even fractions as default
        self.fracs = [1.0 / world_size for i in range(world_size)]
        if fracs:
            self.fracs = fracs

        self.partions = []

        print(len(self.X),len(self.X_eval))
        current_size = 0
        indexes = [i for i in range(0,len(self.X))]
        for frac in self.fracs:
            part_len = int(frac * len(self.X))
            self.partions.append((self.X[indexes[current_size: current_size + part_len]], self.y[indexes[current_size: current_size + part_len]]))
            current_size += part_len

    def get_partion(self,rank):
        print(len(self.partions[rank][0]))
        return TensorDataset(self.partions[rank][0],self.partions[rank][1])
    
    def get_eval(self):
        return TensorDataset(self.X_eval,self.y_eval)
    
def reduce_and_average_gradients(model, world_size):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

def run(rank,world_size,args):
    setup(rank, world_size) # initilize processes
    torch.manual_seed(42)
    model = SimpleNet().to(device)
    dataloader = DataLoader(args['partions'].get_partion(rank),batch_size=args['batch_size'],shuffle=args.get('shuffle',True))
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    num_epochs = args['epochs']
    criterion = nn.CrossEntropyLoss()
    print(f"\n Rank {rank} len train {len(dataloader)}")
    # Training loop
    model.train() 
    dist.barrier()
    for epoch in range(num_epochs):
        # check eval set beginning of each epoch to validate models are scync
        for batch_X, batch_y in dataloader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # average_gradients_wrong(model,world_size)
            reduce_and_average_gradients(model, world_size)
            
            # Update weights
            optimizer.step()
        # eval loss 
        model.eval()
        eval_loss = 0.0
        test_loader = DataLoader(args['partions'].get_eval(),batch_size=args['batch_size'],shuffle=False)
        with torch.no_grad():  # Disable gradient calculation during evaluation
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * inputs.shape[0]

        print(f"\n Rank {rank} Epoch {epoch + 1}, Epoch Loss: {eval_loss / len(test_loader):.4f}")

if __name__ == "__main__":
    world_size = 4 # 4 processes
    args = {}
    args['epochs'] = 2
    args['batch_size'] = 13
    args['partions'] = Partions(X=X,y=y,X_eval=X_eval, y_eval=y_eval, world_size=world_size)

    mp.spawn(fn=run,args=(world_size,args, ), nprocs=world_size, join=True)

