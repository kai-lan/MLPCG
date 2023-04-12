'''
File: model.py
File Created: Tuesday, 10th January 2023 12:51:41 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
--------------
'''
from math import inf, log2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class InvModel(nn.Module):
    def __init__(self):
        super(InvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(8, 1, kernel_size=1, padding='same')
    def forward(self, x):
        x = self.conv1
    def move_to(self, device):
        self.device = device
        self.to(device)
    def loss(self, A_invs, As, fluids):
        r = torch.zeros(1).to(self.device)
        for i in range(As.shape[0]):
            A_inv = A_invs[i]
            A_inv = A_inv.to_sparse_coo()
            A = As[i]
            A = A.to_sparse_coo()
            diag = torch.zeros(A.shape[0], device=self.device)
            diag[fluids] = 1.0
            I = torch.sparse.spdiags(diag, torch.tensor([0]), A.shape)
            r += (A @ A_inv - I).norm()


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
    def __init__(self, ks=3):
        super(FluidNet, self).__init__()
        # self.num_upsamples = log2()
        self.normalizeInputThreshold=0.00001
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
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x_in):
        # Input: x = [bs, 2, Nx, Ny]
        # Normalization might improve invariance of scaling for the network because y = Ax <=> sy = A(sx)
        # Also spead out values to both positive and negative, otherwise relu zero out all negative values
        x = x_in.clone() # Do not modify x
        # std = torch.std(x[:, 0:1], dim=(2, 3), keepdim=True)
        # scale = torch.clamp(std, self.normalizeInputThreshold, inf)
        # x[:, 0:1] /= scale
        x = F.relu(self.conv1(x))
        x1 = self.downsample(x)
        x2 = self.downsample(x1)

        x = F.relu(self.conv2(x))
        x1 = F.relu(self.down11(x1))
        x2 = F.relu(self.down21(x2))

        x = F.relu(self.conv3(x))
        x1 = F.relu(self.down12(x1))
        x2 = F.relu(self.down22(x2))

        x = F.relu(self.conv4(x))
        x1 = F.relu(self.down13(x1))
        x2 = F.relu(self.down23(x2))
        x = x + self.upsample(x1 + self.upsample(x2))

        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        # x = x * scale # In-place operation like x *= 2 or x[:]= ... is not allowed for x that requires auto grad
        ## zero out entries in null space (non-fluid)
        x.masked_fill_(abs(x_in[:, 1:] - 2) > 1e-12, 0) # 2 is fluid cell
        return x

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

    x = torch.stack([rhs, flags]).reshape(1, 2, N, N)

    # model = FluidNet(ks=3)
    # model = DCDM(2)
    model = NewModel1()
    model.eval()
    torch.set_grad_enabled(False) # disable autograd globally
    model.to(torch.device("cuda"))

    summary(model, (2, 256, 256))
    # y = model(x)
    # print(y.shape)