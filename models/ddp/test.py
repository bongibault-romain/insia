import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    print(f"Using GPU {device_id}")
    model = ToyModel().to(device_id)
    print(f"Model created on rank {rank}.")
    ddp_model = DDP(model, device_ids=[device_id])
    print(f"DDP model created on rank {rank}.")
    loss_fn = nn.MSELoss()
    print(f"Loss function created on rank {rank}.")
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print(f"Optimizer created on rank {rank}.")

    optimizer.zero_grad()
    print(f"Optimizer zeroed on rank {rank}.")
    outputs = ddp_model(torch.randn(20, 10))
    print(f"Outputs created on rank {rank}.")
    labels = torch.randn(20, 5).to(device_id)
    print(f"Labels created on rank {rank}.")
    loss_fn(outputs, labels).backward()
    print(f"Loss backwarded on rank {rank}.")
    optimizer.step()
    print(f"Optimizer stepped on rank {rank}.")
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    demo_basic()