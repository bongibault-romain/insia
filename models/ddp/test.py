import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# A simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()  # Get the rank of the process
    world_size = dist.get_world_size()  # Total number of processes (workers)

    print(f"1Rank {rank} is ready to go!")

    # Set device for the current process
    torch.cuda.set_device(rank)
    
    print(f"2Rank {rank} is ready to go!")

    # Load the dataset with distributed sampler
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)

    print(f"3Rank {rank} is ready to go!")

    # Create model and move it to the appropriate device
    model = SimpleModel().cuda()

    print(f"4Rank {rank} is ready to go!")

    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    print(f"5Rank {rank} is ready to go!")

    # # Optimizer and loss function
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # criterion = nn.CrossEntropyLoss()

    # # Training loop
    # for epoch in range(10):
    #     model.train()
    #     train_sampler.set_epoch(epoch)  # Shuffle data for each epoch
    #     running_loss = 0.0
    #     for inputs, targets in train_loader:
    #         inputs, targets = inputs.cuda(rank), targets.cuda(rank)
    #         optimizer.zero_grad()

    #         outputs = model(inputs.view(inputs.size(0), -1))
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #     if rank == 0:  # Print only from the master process
    #         print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    dist.barrier()  # Synchronize all processes before ending

if __name__ == "__main__":
    main()
