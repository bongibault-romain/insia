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
    """Initialise le processus distribué"""
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Process {rank} initialized (Total: {world_size})")

def cleanup():
    """Détruit le groupe de processus distribué"""
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Rank {rank} is running.")

    """Entraînement avec DistributedDataParallel"""
    setup()

    print(f"Rank {rank} has started training.")
    
    # Récupération du rank après initialisation
    rank = dist.get_rank()

    print(f"Running DDP on rank {rank}, using GPU {torch.cuda.current_device()}.")
    
    # Initialisation du modèle sur le bon GPU
    model = ToyModel().to(rank)

    print(f"Model initialized on rank {rank}.")

    model = DDP(model, device_ids=[rank])

    print(f"Model initialized on rank {rank}.")

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # loss_fn = nn.MSELoss()

    # for epoch in range(5):
    #     inputs = torch.randn(16, 10).to(rank)
    #     labels = torch.randn(16, 5).to(rank)

    #     optimizer.zero_grad()
    #     outputs = model(inputs)
    #     loss = loss_fn(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    print(f"Rank {rank} has finished training.")

    cleanup()
    print(f"Rank {rank} has cleaned up.")

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])  # Définir via torchrun
    print(f"Running on {world_size} GPUs.")
    
    mp.spawn(train, args=(world_size,), nprocs=torch.cuda.device_count(), join=True)
