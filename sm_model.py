import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
import numpy as np

class SmallSMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.KL = nn.Conv2d(1, 9, kernel_size=3, padding='same')
    def forward(self, image, x): # 1 x N x N, bs x 1 x N x N
        K = self.KL(image) # 1 x N x N -> 9 x N x N
        K = K.permute((1, 2, 0)) # 9 x N x N -> N x N x 9
        K = K.unflatten(2, (3, 3)) # N x N x 9 -> N x N x 3 x 3
        x = F.pad(x, (1, 1, 1, 1)) # bs x 1 x N x N -> bs x 1 x (N+2) x (N+2)
        x = x.unfold(2, 3, 1).unfold(3, 3, 1) # bs x 1 x (N+2) x (N+2) -> bs x 1 x N x N x 3 x 3
        y = (x * K).sum(dim=(-2, -1)) # bs x 1 x N x N x 3 x 3, N x N x 3 x 3 -> bs x 1 x N x N
        return y

# _____       _____
#      |     |
#      |_____|
class SmallSMModelLegacy(BaseModel):
    def __init__(self):
        super().__init__()
        self.L0 = SmallSMBlock()
        self.L10 = SmallSMBlock()
        self.L11 = SmallSMBlock()
        self.c00 = nn.Conv2d(1, 1, kernel_size=1)
        self.c01 = nn.Conv2d(1, 1, kernel_size=1)
    def forward(self, image, b):
        b0 = self.L0(image, b)
        b0 = F.elu(b0)

        b1 = F.avg_pool2d(b0, (2, 2))
        image0 = F.max_pool2d(image, (2, 2))
        b1 = self.L10(image0, b1)
        b1 = F.elu(b1)

        b1 = F.interpolate(b1, scale_factor=2)
        b1 = self.L11(image, b1)
        b1 = F.elu(b1)
        x = self.c00(b0) + self.c01(b1)
        return x

class SmallLinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.KL = nn.Conv2d(1, 9, kernel_size=3, padding='same')
    def forward(self, image):
        K = self.KL(image) # 1 x N x N -> 9 x N x N
        return K.mean()

# ----pre0----          ----post1----
#            |          |
#            ----pre1----
class SmallSMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.pre0 = SmallSMBlock()
        self.post1 = SmallSMBlock()
        self.pre1 = SmallSMBlock()
        self.c00 = SmallLinearBlock()
        self.c01 = SmallLinearBlock()
    def forward(self, image, b):
        b0 = self.pre0(image, b)
        b0 = F.elu(b0)

        b1 = F.avg_pool2d(b0, (2, 2))
        image1 = F.max_pool2d(image, (2, 2))
        b1 = self.post1(image1, b1)
        b1 = F.elu(b1)

        b1 = F.interpolate(b1, scale_factor=2)
        b1 = self.pre1(image, b1)
        b1 = F.elu(b1)

        x = self.c00(image) * b0 + self.c01(image) * b1
        return x

# _____                   _____
#      |                 |
#      |_____       _____|
#            |     |
#            |_____|
class SmallSMModelD2(BaseModel):
    def __init__(self):
        super().__init__()
        self.pre0 = SmallSMBlock()
        self.post0 = SmallSMBlock()
        self.pre1 = SmallSMBlock()
        self.post1 = SmallSMBlock()
        self.l2 = SmallSMBlock()

        self.c00 = SmallLinearBlock()
        self.c01 = SmallLinearBlock()
        self.c10 = SmallLinearBlock()
        self.c11 = SmallLinearBlock()

    def forward(self, image, b):
        b0 = self.pre0(image, b)
        b0 = F.elu(b0)

        b1 = F.avg_pool2d(b0, (2, 2))
        image1 = F.max_pool2d(image, (2, 2))
        b1 = self.pre1(image1, b1)
        b1 = F.elu(b1)

        b2 = F.avg_pool2d(b1, (2, 2))
        image2 = F.max_pool2d(image1, (2, 2))
        b2 = self.l2(image2, b2)
        b2 = F.elu(b2)

        b2 = F.interpolate(b2, scale_factor=2)
        b2 = self.post1(image1, b2)
        b2 = F.elu(b2)
        b1 = self.c10(image1) * b1 + self.c11(image1) * b2

        b1 = F.interpolate(b1, scale_factor=2)
        b1 = self.post0(image, b1)
        b1 = F.elu(b1)
        b0 = self.c00(image) * b0 + self.c01(image) * b1

        return b0

# _____                               _____
#      |                             |
#      |_____                   _____|
#            |                 |
#            |_____       _____|
#                  |     |
#                  |_____|
class SmallSMModelD3(BaseModel):
    def __init__(self):
        super().__init__()
        self.pre0 = SmallSMBlock()
        self.post0 = SmallSMBlock()
        self.pre1 = SmallSMBlock()
        self.post1 = SmallSMBlock()
        self.pre2 = SmallSMBlock()
        self.post2 = SmallSMBlock()
        self.l3 = SmallSMBlock()

        self.c00 = SmallLinearBlock()
        self.c01 = SmallLinearBlock()
        self.c10 = SmallLinearBlock()
        self.c11 = SmallLinearBlock()
        self.c20 = SmallLinearBlock()
        self.c21 = SmallLinearBlock()

    def forward(self, image, b):
        b0 = self.pre0(image, b)
        b0 = F.elu(b0)

        b1 = F.avg_pool2d(b0, (2, 2))
        image1 = F.max_pool2d(image, (2, 2))
        b1 = self.pre1(image1, b1)
        b1 = F.elu(b1)

        b2 = F.avg_pool2d(b1, (2, 2))
        image2 = F.max_pool2d(image1, (2, 2))
        b2 = self.pre2(image2, b2)
        b2 = F.elu(b2)

        b3 = F.avg_pool2d(b2, (2, 2))
        image3 = F.max_pool2d(image2, (2, 2))
        b3 = self.l3(image3, b3)
        b3 = F.elu(b3)

        b3 = F.interpolate(b3, scale_factor=2)
        b3 = self.post2(image2, b3)
        b3 = F.elu(b3)
        b2 = self.c20(image2) * b2 + self.c21(image2) * b3

        b2 = F.interpolate(b2, scale_factor=2)
        b2 = self.post1(image1, b2)
        b2 = F.elu(b2)
        b1 = self.c10(image1) * b1 + self.c11(image1) * b2

        b1 = F.interpolate(b1, scale_factor=2)
        b1 = self.post0(image, b1)
        b1 = F.elu(b1)
        b0 = self.c00(image) * b0 + self.c01(image) * b1

        return b0

# _____                                           _____
#      |                                         |
#      |_____                               _____|
#            |                             |
#            |_____                   _____|
#                  |                 |
#                  |_____............|
class SmallSMModelDn(BaseModel):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList()
        self.post = nn.ModuleList()
        for _ in range(n):
            self.pre.append(SmallSMBlock())
            self.post.append(SmallSMBlock())
        self.l = SmallSMBlock()
        self.c0 = nn.ModuleList()
        self.c1 = nn.ModuleList()
        for _ in range(n):
            self.c0.append(SmallLinearBlock())
            self.c1.append(SmallLinearBlock())

    def forward(self, image, b):
        x = [F.elu(self.pre[0](image, b))]
        imgs = [image]

        for i in range(1, self.n):
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.max_pool2d(imgs[-1], (2, 2)))
            x[-1] = F.elu(self.pre[i](imgs[-1], x[-1]))

        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.max_pool2d(imgs[-1], (2, 2)))
        x[-1] = F.elu(self.l(imgs[-1], x[-1]))

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = F.elu(self.post[i-1](imgs[i-1], x[i]))
            x[i-1] = self.c0[i-1](imgs[i-1]) * x[i-1] + self.c1[i-1](imgs[i-1]) * x[i]

        return x[0]


class SMConvBlock(nn.Module):
    def __init__(self, nc_in, nc_out, kernel_size=3):
        super().__init__()
        self.nc_in = nc_in
        self.nc_out = nc_out
        self.ks = kernel_size
        self.flat = lambda i, j: nc_in * i + j
        self.KLs = nn.ModuleList()
        for _ in range(nc_in * nc_out):
            self.KLs.append(
                nn.Sequential(nn.Conv2d(1, 20, kernel_size=kernel_size, padding='same'),
                              nn.Conv2d(20, kernel_size**2, kernel_size=kernel_size, padding='same'))# image -> kernel/filter
            )

    def forward(self, image, x): # image: bs x N x N, x: bs x nc x N x N
        bs, _, N, _ = x.shape
        y = torch.zeros(bs, self.nc_out, N, N, device=x.device)
        # print('before', torch.cuda.memory_allocated(0))
        for i in range(self.nc_out):
            for j in range(self.nc_in):
                K = self.KLs[self.flat(i, j)](image.unsqueeze(1)) # 1 x N x N -> 1 x 1 x N x N -> 1 x 9 x N x N
                K = K.permute((0, 2, 3, 1)) # 1 x 9 x N x N -> 1 x N x N x 9
                K = K.unflatten(3, (3, 3)) # 1 x N x N x 9 -> 1 x N x N x 3 x 3
                x_unf = F.pad(x[:, j], (1, 1, 1, 1)) # bs x N x N -> bs x (N+2) x (N+2). 'Pad' starts from the last dim...
                x_unf = x_unf.unfold(1, 3, 1).unfold(2, 3, 1) # bs x (N+2) x (N+2) -> bs x N x N x 3 x 3
                y[:, i] += (x_unf * K).sum(dim=(3, 4)) # (bs x N x N x 3 x 3, 1 x N x N x 3 x 3) -> bs x N x N
        # print('after', torch.cuda.memory_allocated(0))
        return y

class SMModelFull(BaseModel):
    def __init__(self):
        super().__init__()
        ks = 3
        nc = 8

        self.init = SMConvBlock(1, nc)

        self.conv01 = SMConvBlock(nc, nc, ks)
        self.conv02 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv03 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv04 = SMConvBlock(nc, nc, kernel_size=ks)

        self.conv11 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv12 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv13 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv14 = SMConvBlock(nc, nc, kernel_size=ks)

        self.conv21 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv22 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv23 = SMConvBlock(nc, nc, kernel_size=ks)
        self.conv24 = SMConvBlock(nc, nc, kernel_size=ks)

        self.conv5 = nn.Conv2d(nc, nc, kernel_size=1, padding='same')
        self.conv6 = nn.Conv2d(nc, 1, kernel_size=1, padding='same')

        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
    def forward(self, image, x): # 1 x N x N, bs x 1 x N x N
        # v = x[:, 0:1].clone()
        # image = x[0:1, 1].clone() # 1 x N x N image the same for all batches
        image1 = self.downsample(image)
        image2 = self.downsample(image1)

        activate = F.elu
        x = activate(self.init(image, x))

        x1 = self.downsample(x)
        x2 = self.downsample(x1)

        x = activate(self.conv01(image, x))
        x1 = activate(self.conv11(image1, x1))
        x2 = activate(self.conv21(image2, x2))


        x = activate(self.conv02(image, x))
        x1 = activate(self.conv12(image1, x1))
        x2 = activate(self.conv22(image2, x2))

        x = activate(self.conv03(image, x))
        x1 = activate(self.conv13(image1, x1))
        x2 = activate(self.conv23(image2, x2))

        x = activate(self.conv04(image, x))
        x1 = activate(self.conv14(image1, x1))
        x2 = activate(self.conv24(image2, x2))

        x1 = self.upsample(x1)
        x2 = self.upsample(self.upsample(x2))

        x = x + x1 + x2

        x = activate(self.conv5(x))
        x = self.conv6(x)
        return x

if __name__ == '__main__':
    import os, sys
    path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(path + "/lib")
    from lib.read_data import *
    import matplotlib.pyplot as plt
    from torchsummary import summary

    N = 256
    frame = 100
    # file_A = os.path.join(path, "data_fluidnet", "dambreak_2D_64", f"A_{frame}.bin")
    file_rhs = os.path.join(DATA_PATH, f"dambreak_N{N}_200", f"div_v_star_{frame}.bin")
    file_sol = os.path.join(DATA_PATH, f"dambreak_N{N}_200", f"pressure_{frame}.bin")
    file_flags = os.path.join(DATA_PATH, f"dambreak_N{N}_200", f"flags_{frame}.bin")
    # A = readA_sparse(64, file_A, DIM=2)
    rhs = torch.tensor(load_vector(file_rhs), dtype=torch.float32)
    flags = torch.tensor(read_flags(file_flags), dtype=torch.float32)
    sol = torch.tensor(load_vector(file_sol), dtype=torch.float32)

    x = torch.stack([rhs, flags]).reshape(1, 2, N, N).to(torch.device('cuda'))

    model = SmallSMModel()
    # model = DCDM(2)
    # model = NewModel1()
    # model.eval()
    torch.set_grad_enabled(False) # disable autograd globally
    model.to(torch.device("cuda"))

    summary(model, ((1, 256, 256), (1, 1, 256, 256)))
    # y = model(flags.reshape(1, N, N), rhs.reshape(1, 1, N, N))
    # loss = model.residual_loss(x, y, A)
    # print(y.shape)
    print(model)