'''
File: model.py
File Created: Tuesday, 10th January 2023 12:51:41 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
--------------
'''
from math import inf, log2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    def move_to(self, device):
        self.device = device
        self.to(device)
    def eval_forward(self, *args, **kargs):
        return self.forward(*args, **kargs)
    def reset_parameters(self, weight, bias):
        torch.manual_seed(0)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)

