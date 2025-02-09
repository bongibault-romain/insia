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

def setup(rank, world_size):
    dist.init_process_group("nccl")  # Backend pour GPU, "gloo" pour CPU
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    print(f"Rank {rank} is using GPU {torch.cuda.current_device()}.")
    model = ToyModel().to(rank)
    print(f"Model on rank {rank} is on GPU {next(model.parameters()).device}.")
    model = DDP(model, device_ids=[rank])
    print(f"Model on rank {rank} is on GPU {next(model.parameters()).device}.")

    # Votre boucle d'entraînement ici...

    print(f"Rank {rank} is done.")
    cleanup()
    print(f"Rank {rank} has cleaned up.")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running on {world_size} GPUs.")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)