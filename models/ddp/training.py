import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import torch.multiprocessing as mp
import time
import model
from constants import *

from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
    # if model_path exists, load it
    if os.path.exists(model_path):
        print('Loading model...')
        model = torch.load(model_path, weights_only=False)
    else:
        print('Creating model...')    
        model = model.GPTLanguageModel().to(device)

    print(f"Model initialized on rank {rank}.")

    model = DDP(model, device_ids=[0])  # Since each node has one GPU, we use device 0

    print(f"Model initialized on rank {rank}.")

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        print(f"step {(iter/max_iters)*100:.2f}% complete ({iter}/{max_iters}).")

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Rank {rank} has finished training.")

    # Wait for all processes to finish

    dist.barrier()

    cleanup()
    print(f"Rank {rank} has cleaned up.")

    # Save the model

    if rank == 0:
        print("Saving model...")
        torch.save(model.state_dict(), "model.pth") 


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])  # Set via torchrun
    rank = int(os.environ["NODE_RANK"])  # Set via environment

    print(f"Running on {world_size} GPUs.")

    # We only have one process per node, so nprocs = 1
    mp.spawn(train, args=(rank, world_size), nprocs=1, join=True)
