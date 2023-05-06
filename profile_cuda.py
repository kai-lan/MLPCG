import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from model import *
import torch.optim as optim

cuda = torch.device('cuda')
mb = 1024**2
x = torch.randn(128, 1, 256, 256)
image = torch.randn(128, 1, 256, 256)
x = x.to(cuda)
image = image.to(cuda)
print('init', torch.cuda.memory_allocated()/mb)

# model = SMConvBlock(1, 16, kernel_size=3)
model = MyConvBlock(2, 16, kernel_size=3)
model.to(cuda)
print("After model to device", torch.cuda.memory_allocated()/mb)

optimizer = optim.Adam(model.parameters())
print("After optimizer init", torch.cuda.memory_allocated()/mb)

for i in range(3):
    print('Iter', i)
    print('Before forward pass', torch.cuda.memory_allocated()/mb)
    # y = model(image.squeeze(), x)
    y = model(torch.cat([x, image], dim=1))
    b = torch.cuda.memory_allocated()
    print('After forward pass', b/mb)
    z = y.sum()
    z.backward()
    print('After backward pass', torch.cuda.memory_allocated()/mb)
    optimizer.step()
    print("After optimizer step", torch.cuda.memory_allocated()/mb)
