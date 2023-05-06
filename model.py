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
from torchsummary import summary



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    def move_to(self, device):
        self.device = device
        self.to(device)
        # Residual loss
    def residual_loss(self, x, y, A):
        bs = x.shape[0]
        r = torch.zeros(1).to(x.device)
        for i in range(bs):
            r += (y[i] - A @ x[i]).norm() # No need to compute relative residual because inputs are all unit vectors
        return r / bs
    # Energy loss: negative decreasing
    def energy_loss(self, x, b, A):
        bs = x.shape[0]
        r = torch.zeros(1).to(x.device)
        for i in range(bs):
            r += 0.5 * x[i].dot(A @ x[i]) - x[i].dot(b[i])
        return r / bs
    # Scaled loss in 2-norm
    def scaled_loss_2(self, x, y, A): # bs x dim x dim (x dim)
        bs = x.shape[0]
        result = torch.zeros(1, dtype=x.dtype, device=x.device)
        for i in range(bs):
            Ax = A @ x[i]
            alpha = x[i].dot(y[i]) / x[i].dot(Ax)
            r = (y[i] - alpha * Ax).square().sum()
            result += r
        return result / bs
    def scaled_loss_A(self, x, y, A): # bs x dim x dim (x dim)
        bs = x.shape[0]
        result = torch.zeros(1, dtype=x.dtype, device=x.device)
        for i in range(bs):
            Ax = A @ x[i]
            alpha = x[i].dot(y[i]) / x[i].dot(Ax)
            r = y[i] - alpha * Ax
            result += r.dot(A @ r)
        return result / bs
# x -> L0 x -> L1 L0 x -> L2 L1 L0 x -> ...
#       |         |           |
#       |_________|___________|
#                 |
#      c0 L0 x + c1 L1 L0 x + c2 L2 L1 L0 x + ...
class LinearModel(BaseModel):
    def __init__(self, levels=1):
        super(LinearModel, self).__init__()
        self.L = nn.ModuleList()
        self.levels = levels
        for _ in range(levels):
            self.L.append(nn.Conv2d(1, 1, kernel_size=3, padding_mode='circular', padding='same', bias=False))
        for _ in range(levels):
            self.L.append(nn.Conv2d(1, 1, kernel_size=1, padding_mode='circular', padding='same', bias=False))
        # self.B0 = nn.Conv2d(1, 1, kernel_size=3, padding_mode='circular', padding='same', bias=False)
        # self.B1 = nn.Conv2d(1, 1, kernel_size=3, padding_mode='circular', padding='same', bias=False)
        # self.C0 = nn.Conv2d(1, 1, kernel_size=1, padding_mode='circular', padding='same', bias=False)
        # self.C1 = nn.Conv2d(1, 1, kernel_size=1, padding_mode='circular', padding='same', bias=False)
    def forward(self, image, x_in):
        N = image.shape[-1]
        bs = x_in.shape[0]
        y = [self.L[0](x_in)]
        for i in range(1, self.levels):
            y.append(self.L[i](y[-1]))
        z = self.L[self.levels](y[0])
        for i in range(1, self.levels):
            z = z + self.L[self.levels+i](y[i])
        # x = self.C0(x0) + self.C1(x1)
        return z

class SimpleModel(BaseModel):
    def __init__(self) -> None:
        super(SimpleModel, self).__init__()
        self.init = nn.Conv2d(2, 12, kernel_size=3, padding='same')
        self.top1 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.top2 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.top3 = nn.Conv2d(12, 12, kernel_size=3, padding='same')


        self.down1 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.down2 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.down3 = nn.Conv2d(12, 12, kernel_size=3, padding='same')


        self.ddown1 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.ddown2 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.ddown3 = nn.Conv2d(12, 12, kernel_size=3, padding='same')


        self.dddown1 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.dddown2 = nn.Conv2d(12, 12, kernel_size=3, padding='same')
        self.dddown3 = nn.Conv2d(12, 12, kernel_size=3, padding='same')


        self.flat  = nn.Conv2d(12, 12, kernel_size=1, padding='same')
        self.last  = nn.Conv2d(12, 1, kernel_size=1, padding='same')
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # x = F.normalize(x, dim=0)
        # x = torch.stack([vec.squeeze(dim=1), geo.squeeze(dim=1)], dim=1)
        x = F.relu(self.init(x))
        x1 = self.downsample(x)
        x2 = self.downsample(x1)
        x3 = self.downsample(x2)

        x = F.relu(self.top1(x))
        x1= F.relu(self.down1(x1))
        x2= F.relu(self.ddown1(x2))
        x3= F.relu(self.dddown1(x3))

        x = F.relu(self.top2(x))
        x1 = F.relu(self.down2(x1))
        x2 = F.relu(self.ddown2(x2))
        x3 = F.relu(self.dddown2(x3))

        x = F.relu(self.top3(x))
        x1 = F.relu(self.down3(x1))
        x2 = F.relu(self.ddown3(x2))
        x3 = F.relu(self.dddown3(x3))

        # x = F.relu(self.top4(x))
        # x1 = F.relu(self.down4(x1))
        # x2 = F.relu(self.ddown4(x2))
        # x3 = F.relu(self.dddown4(x3))

        x = x + self.upsample(x1) + self.upsample(self.upsample(x2)) + self.upsample(self.upsample(self.upsample(x3)))
        x = F.relu(self.flat(x))
        x = self.last(x)
        return x

class FluidNet(BaseModel):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, ks=3, depth=2, length=3):
        super(FluidNet, self).__init__()
        self.ks = ks
        self.depth = depth
        self.length = length
        self.flat_ind = lambda i, j: i * (length+1) + j
        nc = 16

        self.init = nn.Conv2d(2, nc, kernel_size=ks, padding='same')

        self.conv01 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv02 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv03 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv04 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')

        self.conv11 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv12 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv13 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv14 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')

        self.conv21 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv22 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv23 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')
        self.conv24 = nn.Conv2d(nc, nc, kernel_size=ks, padding='same')

        self.conv5 = nn.Conv2d(nc, nc, kernel_size=1, padding='same')
        self.conv6 = nn.Conv2d(nc, 1, kernel_size=1, padding='same')

        # self.downsample = nn.Conv2d(nc, nc, kernel_size=2, stride=2)
        # self.upsample = nn.ConvTranspose2d(nc, nc, kernel_size=2, stride=2)
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, image, x_in):
        N = image.shape[-1]
        bs = x_in.shape[0]
        x = torch.cat([x_in, image.view(1, 1, N, N).expand(bs, 1, N, N)], dim=1)
        # x = x_in.clone()
        # std = torch.std(x[:, 0:1], dim=(2, 3), keepdim=True)
        # scale = torch.clamp(std, 0.00001, inf)
        # x[:, 0:1] /= scale
        activate = F.elu
        x = activate(self.init(x))

        x1 = self.downsample(x)
        x2 = self.downsample(x1)

        x = activate(self.conv01(x))
        x1 = activate(self.conv11(x1))
        x2 = activate(self.conv21(x2))


        x = activate(self.conv02(x))
        x1 = activate(self.conv12(x1))
        x2 = activate(self.conv22(x2))

        x = activate(self.conv03(x))
        x1 = activate(self.conv13(x1))
        x2 = activate(self.conv23(x2))

        x = activate(self.conv04(x))
        x1 = activate(self.conv14(x1))
        x2 = activate(self.conv24(x2))

        x1 = self.upsample(x1)
        x2 = self.upsample(self.upsample(x2))


        x = x + x1 + x2

        x = activate(self.conv5(x))
        x = self.conv6(x)
        # x = x * scale # In-place operation like x *= 2 or x[:]= ... is not allowed for x that requires auto grad
        ## zero out entries in null space (non-fluid)
        x.masked_fill_(abs(image - 2) > 1e-12, 0) # 2 is fluid cell
        return x
class MyConvBlock(nn.Module):
    def __init__(self, nc_in, nc_out, kernel_size=3):
        super().__init__()
        self.nc_in = nc_in
        self.nc_out = nc_out
        self.ks = kernel_size
        self.w = nn.Parameter(torch.randn(nc_out, nc_in, kernel_size, kernel_size)) # nc_out x nc x 3 x 3
    def forward(self, x): # bs x nc x N x N
        bs, _, N, _ = x.shape
        # y = F.conv2d(x, self.w)
        # y = torch.zeros(bs, self.nc_out, N, N, device=x.device)
        # for i in range(self.nc_out):
        #     for j in range(self.nc_in):
        #         x_unf = F.pad(x[:, j], (1, 1, 1, 1)) # bs x N x N -> bs x (N+2) x (N+2). 'Pad' starts from the last dim...
        #         x_unf = x_unf.unfold(1, 3, 1).unfold(2, 3, 1) # bs x (N+2) x (N+2) -> bs x N x N x 3 x 3
        #         y[:, i] += (x_unf * self.w[i][j]).sum(dim=(3, 4))
        # x_unf = F.unfold(x, (self.ks, self.ks)) # bs x nc x N x N -> bs x (nc x 3 x 3) x ((N-2) x (N-2))
        # x_unf = x_unf.permute((0, 2, 1)) # bs x (nc x 3 x 3) x ((N-2) x (N-2)) -> bs x ((N-2) x (N-2)) x (nc x 3 x 3)
        # y = x_unf @ (self.w.view(self.w.size(0), -1).t()) # bs x ((N-2) x (N-2)) x (nc x 3 x 3), (nc x 3 x 3) x nc_out -> bs x ((N-2) x (N-2)) x nc_out
        # y = y.permute((0, 2, 1)) # bs x ((N-2) x (N-2)) x nc_out -> bs x nc_out x ((N-2) x (N-2))
        # # y = y.unflatten(2, (N-2, N-2))
        # y = y.view(bs, self.nc_out, N-2, N-2)

        y1 = (F.unfold(x, (self.ks, self.ks)).transpose(1, 2) @ (self.w.view(self.w.size(0), -1).t())).transpose(1, 2).view(bs, self.nc_out, N-2, N-2)
        # y2 = F.conv2d(x, self.w)
        # print((y1 - y2).abs().max())
        return y1

class SmallSMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.KL = nn.Conv2d(1, 9, kernel_size=3, padding='same')
    def forward(self, image, x): # 1 x N x N, bs x 1 x N x N
        K = self.KL(image) # 1 x N x N -> 9 x N x N
        K = K.permute((1, 2, 0)) # 9 x N x N -> N x N x 9
        K = K.unflatten(2, (3, 3)) # N x N x 9 -> N x N x 3 x 3
        # print(K[0, 1])
        x = F.pad(x.squeeze(1), (1, 1, 1, 1)) # bs x 1 x N x N -> bs x (N+2) x (N+2)
        x = x.unfold(1, 3, 1).unfold(2, 3, 1) # bs x (N+2) x (N+2) -> bs x N x N x 3 x 3
        y = (x * K).sum(dim=(3, 4)) # bs x N x N x 3 x 3, N x N x 3 x 3 -> bs x N x N
        return y

class SmallSMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.L0 = SmallSMBlock()
        self.L1 = SmallSMBlock()
        self.L2 = SmallSMBlock()
        self.c0 = nn.Conv2d(1, 1, kernel_size=1)
        self.c1 = nn.Conv2d(1, 1, kernel_size=1)
    def forward(self, image, b):
        b0 = self.L0(image, b)
        # b0 = F.elu(b0)
        p0 = F.avg_pool2d(b0, (2, 2))
        image0 = F.max_pool2d(image, (2, 2))
        p1 = self.L1(image0, p0)
        u1 = F.upsample(p1.unsqueeze(1), scale_factor=2)
        b1 = self.L2(image, u1)
        x = self.c0(b0.unsqueeze(1)) + self.c1(b1.unsqueeze(1))
        return x

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
class SMModel(BaseModel):
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

    def loss(self, x, b, A):
        bs = x.shape[0]
        r = torch.zeros(1).to(x.device)
        for i in range(bs):
            r += (b[i] - A @ x[i]).norm() # No need to compute relative residual because inputs are all unit vectors
        return r / bs


class NewModel(FluidNet): # Made downsampling and upsampling learnable
    def __init__(self):
        super(NewModel, self).__init__()
        self.downsample = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1)

class NewModel1(BaseModel): # increase num of channels as we downsample
    def __init__(self, ks=3):
        super(NewModel1, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')

        self.conv4 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.conv5 = nn.Conv2d(16, 16, kernel_size=1, padding='same', padding_mode='zeros')
        self.conv6 = nn.Conv2d(16, 1, kernel_size=1, padding='same', padding_mode='zeros')

        self.down11 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.down12 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.down13 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')

        self.down21 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.down22 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')
        self.down23 = nn.Conv2d(16, 16, kernel_size=ks, padding='same', padding_mode='zeros')

        self.downsample1 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        self.downsample2 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
    def forward(self, x_in):
        x = x_in.clone()
        # std = torch.std(x[:, 0:1], dim=(2, 3), keepdim=True)
        # scale = torch.clamp(std, self.normalizeInputThreshold, inf)
        # x[:, 0:1] /= scale

        x = F.relu(self.conv1(x))
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)

        x = F.relu(self.conv2(x))
        x1 = F.relu(self.down11(x1))
        x2 = F.relu(self.down21(x2))

        x = F.relu(self.conv3(x))
        x1 = F.relu(self.down12(x1))
        x2 = F.relu(self.down22(x2))

        x = F.relu(self.conv4(x))
        x1 = F.relu(self.down13(x1))
        x2 = F.relu(self.down23(x2))
        x = x + self.upsample1(x1 + self.upsample2(x2))

        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        # x = x * scale
        x.masked_fill_(abs(x_in[:, 1:] - 2) > 1e-12, 0)
        return x

class DCDM(nn.Module):
    def __init__(self, DIM):
        super(DCDM, self).__init__()
        self.DIM = DIM
        Conv = eval(f"nn.Conv{DIM}d")
        AvgPool = eval(f"nn.AvgPool{DIM}d")
        self.cnn1 = Conv(1, 16, kernel_size=3, padding='same')
        self.cnn2 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn3 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn4 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn5 = Conv(16, 16, kernel_size=3, padding='same')

        self.downsample = AvgPool(2)
        self.down1 = Conv(16, 16, kernel_size=3, padding='same')
        self.down2 = Conv(16, 16, kernel_size=3, padding='same')
        self.down3 = Conv(16, 16, kernel_size=3, padding='same')
        self.down4 = Conv(16, 16, kernel_size=3, padding='same')
        self.down5 = Conv(16, 16, kernel_size=3, padding='same')
        self.down6 = Conv(16, 16, kernel_size=3, padding='same')
        self.upsample = nn.Upsample(scale_factor=2)

        self.cnn6 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn7 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn8 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn9 = Conv(16, 16, kernel_size=3, padding='same')
        self.linear = Conv(16, 1, kernel_size=1) # this acts as a linear layer across channels
    def forward(self, x): # shape: bs x nc x dim x dim (x dim)
        first_layer = self.cnn1(x)
        la = F.relu(self.cnn2(first_layer))
        lb = F.relu(self.cnn3(la))
        la = F.relu(self.cnn4(lb) + la)
        lb = F.relu(self.cnn5(la))

        apa = self.downsample(lb)
        apb = F.relu(self.down1(apa))
        apa = F.relu(self.down2(apb) + apa)
        apb = F.relu(self.down3(apa))
        apa = F.relu(self.down4(apb) + apa)
        apb = F.relu(self.down5(apa))
        apa = F.relu(self.down6(apb) + apa)

        upa = self.upsample(apa) + lb
        upb = F.relu(self.cnn6(upa))
        upa = F.relu(self.cnn7(upb) + upa)
        upb = F.relu(self.cnn8(upa))
        upa = F.relu(self.cnn9(upb) + upa)
        last_layer = self.linear(upa)
        return last_layer
    def loss(self, y_pred, y_true, A_sparse): # bs x dim x dim (x dim)
        ''' y_true: r, (bs, N)
        y_pred: A_hat^-1 r, (bs, N)
        '''
        y_pred = y_pred.flatten(1) # Keep bs
        y_true = y_true.flatten(1)
        YhatY = (y_true * y_pred).sum(dim=1) # Y^hat * Y, (bs,)
        YhatAt = (A_sparse @ y_pred.T).T # y_pred @ A_sparse.T not working, Y^hat A^T, (bs, N)
        YhatYhatAt = (y_pred * YhatAt).sum(dim=1) # Y^hat * (Yhat A^T), (bs,)
        return (y_true - torch.diag(YhatY/YhatYhatAt) @ YhatAt).square().sum(dim=1).mean() # /bs / N
    def move_to(self, device):
        self.device = device
        self.to(device)

# SteadyFlowNet https://dl.acm.org/doi/pdf/10.1145/2939672.2939738
class SteadyFlowNet(BaseModel):
    def __init__(self):
        super(SteadyFlowNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=8, stride=8) # 2 x 256 x 256 -> 64 x 32 x 32
        self.conv2 = nn.Conv2d(64, 256, kernel_size=4, stride=4) # 64 x 32 x 32 -> 256 x 8 x 8
        self.conv3 = nn.Conv2d(256, 256, kernel_size=8, stride=8) # 256 x 8 x 8 -> 256 x 1 x 1
        self.dcon1 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=8) # 256 x 1 x 1 -> 256 x 8 x 8
        self.dcon2 = nn.ConvTranspose2d(256, 128, kernel_size=8, stride=8) # 256 x 8 x 8 -> 128 x 64 x 64
        self.dcon3 = nn.ConvTranspose2d(128, 16, kernel_size=2, stride=2) # 128 x 64 x 64 -> 16 x 128 x 128
        self.dcon4 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2) # 16 x 128 x 128 -> 1 x 256 x 256
    def forward(self, x_in):
        x = x_in.clone()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # x = x.flatten(1)
        x = self.dcon1(x)
        x = F.relu(x)
        x = self.dcon2(x)
        x = F.relu(x)
        x = self.dcon3(x)
        x = F.relu(x)
        x = self.dcon4(x)
        x.masked_fill_(abs(x_in[:, 1:] - 2) > 1e-12, 0)
        return x
if __name__ == '__main__':
    import os, sys
    path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(path + "/lib")
    from lib.read_data import *
    import matplotlib.pyplot as plt

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

    # model = FluidNet()
    model = SmallSMModel()
    # model = DCDM(2)
    # model = NewModel1()
    # model.eval()
    # torch.set_grad_enabled(False) # disable autograd globally
    model.to(torch.device("cuda"))

    summary(model, ((1, 256, 256), (2, 256, 256)))
    # y = model(flags.reshape(1, N, N), rhs.reshape(1, 1, N, N))
    # loss = model.residual_loss(x, y, A)
    # print(y.shape)
    print(model)