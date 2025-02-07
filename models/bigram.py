import os
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(250603)

# ------ Constants ------
block_size = 128
batch_size = 64
max_iters = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
model_path = './models/bigram.pth'
data_path = './data/input.txt'

print('Device:', device)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read().strip()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.8) # 80% of the data for training
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

@torch.no_grad()
def estimated_loss():
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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #(B (batch), T (time), C (channel))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits = logits[:, -1, :] 

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx

# if model_path exists, load it
if os.path.exists(model_path):
    print('Loading model...')
    model = torch.load(model_path, weights_only=False)
else:
    print('Creating model...')
    model = BigramLanguageModel(vocab_size)

m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(max_iters):
    xb, yb = get_batch('train')

    if steps % eval_interval == 0:
        losses = estimated_loss()
        print(f'Step: {steps}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

torch.save(m, './models/bigram.pth')

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))