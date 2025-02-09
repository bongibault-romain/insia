import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from constants import *
from tokenization import *
from model import *

model_path_user = input("Enter model path: ")

# if model_path exists, load it
if os.path.exists(model_path_user):
    print('Loading model...')
    model = torch.load(model_path_user, weights_only=False)
else:
    print("model_path does not exist")
    exit()

model = model.to(device)

# ask to the user to enter maximum number of tokens
max_tokens = int(input("Enter maximum number of tokens: "))
prompt = input("Enter prompt: ")

# encode the prompt
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

# generate from the model
print(decode(model.generate(context, max_new_tokens=max_tokens)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))