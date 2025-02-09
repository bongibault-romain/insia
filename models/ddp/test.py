import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup():
    """Initialize the distributed process group"""
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())  # Ensure each rank gets a valid GPU
    print(f"Process {rank} initialized (Total: {world_size})")

def cleanup():
    """Destroy the distributed process group"""
    dist.destroy_process_group()

def train(i, rank, world_size):
    print(f"World size: {world_size}.")
    print(f"Rank: {rank}.")
    print(f"Process Index {i}.")

    print(f"Rank {rank} is running.")

    """Training with DistributedDataParallel"""
    setup()

    print(f"Rank {rank} has started training.")
    
    # Get rank after initialization
    rank = dist.get_rank()

    print(f"Running DDP on rank {rank}, using GPU {torch.cuda.current_device()}.")
    
    # Initialize model on the correct GPU (cuda:0)
    device = torch.device(f"cuda:0")
    model = ToyModel().to(device)

    print(f"Model initialized on rank {rank}.")

    model = DDP(model, device_ids=[0])  # Since each node has one GPU, we use device 0

    print(f"Model initialized on rank {rank}.")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print(f"Rank {rank} has initialized the model.")

    for epoch in range(2):
        print(f"Rank {rank} is training on epoch {epoch}.")
        inputs = torch.randn(16, 10).to(device)
        labels = torch.randn(16, 5).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        print(f"Rank {rank} has computed loss.")

        loss.backward()
        optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    print(f"Rank {rank} has finished training.")

    cleanup()
    print(f"Rank {rank} has cleaned up.")

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])  # Set via torchrun
    rank = int(os.environ["NODE_RANK"])  # Set via environment

    print(f"Running on {world_size} GPUs.")

    # We only have one process per node, so nprocs = 1
    mp.spawn(train, args=(rank, world_size), nprocs=1, join=True)
