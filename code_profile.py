import torch
import torchvision.models as models
import torch.autograd.profiler as profiler
from sm_model import *
# from torch_extension.main import SmallSMBlock
# from pytorch_memlab import profile, set_target_gpu


# @profile
# def f():

model = SmallSMModelDn3D(3).cuda()
# model = nn.Conv2d(1, 9, kernel_size=3, padding='same')
image = torch.randint(2, 4, (1, 64, 64, 64)).float().cuda()
x = torch.randn(1, 1, 64, 64, 64, requires_grad=True).cuda()

# f()
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(inputs[0])
y = model(image, x)

with profiler.profile(use_cuda=True, with_stack=True, profile_memory=True) as prof:
    for i in range(100):
        y = model(image, x)
    y.sum().backward()
    torch.cuda.synchronize()

import pandas as pd
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", top_level_events_only=False, max_shapes_column_width=50))
# df = pd.DataFrame({e.key:e.__dict__ for e in prof.key_averages()}).T
# print(df[['count', 'cuda_time_total', 'self_cuda_time_total', 'cuda_memory_usage', 'self_cuda_memory_usage']].sort_values(['cuda_time_total', 'cuda_memory_usage'], ascending=False))
# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))