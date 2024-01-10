import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
import math

from lib.GLOBAL_VARS import *
from lib.global_clock import *

class SmallSMBlock3DPY(BaseModel):
    def __init__(self, num_imgs=3):
        super().__init__()
        self.KL = nn.Conv3d(num_imgs, 27, kernel_size=3, padding='same', bias=True)
        self.reset_parameters(self.KL.weight, self.KL.bias)
    def forward(self, image, x): # 1 x N x N x N, bs x 1 x N x N x N
        K = self.KL(image) # 1 x N x N x N -> 27 x N x N x N
        K = K.permute((1, 2, 3, 0)) # 27 x N x N x N -> N x N x N x 27
        K = K.unflatten(3, (3, 3, 3)) # N x N x N x 27 -> N x N x N x 3 x 3 x 3
        x = F.pad(x, (1, 1, 1, 1, 1, 1)) # bs x 1 x N x N x N -> bs x 1 x (N+2) x (N+2) x (N+2)
        x = x.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1) # bs x 1 x (N+2) x (N+2) x (N+2) -> bs x 1 x N x N x N x 3 x 3 x 3
        y = (x * K).sum(dim=(-3, -2, -1)) # bs x 1 x N x N x N x 3 x 3 x 3, N x N x N x 3 x 3 x 3 -> bs x 1 x N x N x N
        return y

class SmallLinearBlock3DPY(BaseModel):
    def __init__(self, num_imgs=3):
        super().__init__()
        self.KL = nn.Conv3d(num_imgs, 27, kernel_size=3, padding='same')
        self.reset_parameters(self.KL.weight, self.KL.bias)
    def forward(self, image):
        K = self.KL(image) # num_imgs x N x N x N -> 27 x N x N x N
        return K.mean()

class SmallSMModelDn3DPY(BaseModel):
    def __init__(self, n, num_imgs=3):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList([SmallSMBlock3DPY(num_imgs) for _ in range(n)])
        self.post = nn.ModuleList([SmallSMBlock3DPY(num_imgs) for _ in range(n)])

        self.l = SmallSMBlock3DPY(num_imgs)

        self.c0 = nn.ModuleList([SmallLinearBlock3DPY(num_imgs) for _ in range(n)])
        self.c1 = nn.ModuleList([SmallLinearBlock3DPY(num_imgs) for _ in range(n)])

    def forward(self, image, b):
        x = [self.pre[0](image, b)]
        imgs = [image]

        for i in range(1, self.n):
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
            x[-1] = self.pre[i](imgs[-1], x[-1])

        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = self.post[i-1](imgs[i-1], x[i])
            x[i-1] = self.c0[i-1](imgs[i-1]) * x[i-1] + self.c1[i-1](imgs[i-1]) * x[i]

        return x[0]
