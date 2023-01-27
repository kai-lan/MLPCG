'''
File: model.py
File Created: Tuesday, 10th January 2023 12:51:41 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
Last Modified: Thursday, 12th January 2023 12:28:31 am
--------------
'''
from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from lib.cell_type import flagsToOccupancy
# from lib.velocity_update import velocityUpdate
# from lib import FlexiUNet, MultiScaleNet, UNet, UNet3D, fluid
from flexiunet import FlexiUNet

class _ScaleNet(nn.Module):
    # normalizeInputThreshold : don't normalize input noise
    def __init__(self, normalizeInputThreshold=0.00001):
        super(_ScaleNet, self).__init__()
        self.normalizeInputThreshold = normalizeInputThreshold
    def forward(self, x):
        bs = x.size(0)
        # Rehaspe form (bs x nc x Nx x Ny) to (bs x -1)
        y = x.view(bs, -1)
        # Calculate std using Bessel's correction (correction with n/n-1)
        std = torch.std(y, dim=1, keepdim=True)  # output is size (bs x 1)
        scale = torch.clamp(std, self.normalizeInputThreshold, inf)
        scale = scale.view(bs, 1, 1, 1)
        return scale


class _HiddenConvBlock(nn.Module):
    def __init__(self, dropout=True):
        super(_HiddenConvBlock, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.ReLU(),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FluidNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self):
        super(FluidNet, self).__init__()
        self.scale = _ScaleNet() # Find std of data
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.conv4 = nn.Conv2d(8, 8, kernel_size=1, padding='same', padding_mode='replicate')
        self.conv5 = nn.Conv2d(8, 1, kernel_size=1, padding='same', padding_mode='replicate')
        self.down11 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.down12 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.down21 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.down22 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.downsample = nn.AvgPool2d(2)
        self.updample = nn.Upsample(scale_factor=2)
    def forward(self, x):
        # Input: x = [bs, 2, Nx, Ny]
        # x[:, 0] = div_v_star
        # x[:, 1] = flags
        div_v_star = x[:, 0:1]
        flags = x[:, 1:2]

        # Normalize
        scale = self.scale(div_v_star)
        div_v_star /= scale

        x = F.relu(self.conv1(x))
        x1 = self.downsample(x)
        x2 = self.downsample(x1)

        x = F.relu(self.conv2(x))
        x1 = F.relu(self.down11(x1))
        x2 = F.relu(self.down21(x2))

        x = F.relu(self.conv3(x))
        x1 = F.relu(self.down12(x1))
        x2 = F.relu(self.down22(x2))
        x += self.updample(x1) + self.updample(self.updample(x2))

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x *= scale
        return x

if __name__ == '__main__':
    import os, sys
    sys.path.append("../lib")
    from read_data import read_flags, load_vector
    path = os.path.dirname(os.path.relpath(__file__))
    frame = 20
    # file_A = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"A_{frame}.bin")
    file_rhs = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"div_v_star_{frame}.bin")
    file_sol = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"pressure_{frame}.bin")
    file_flags = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"flags_{frame}.bin")
    # A = readA_sparse(64, file_A, DIM=2)
    rhs = torch.tensor(load_vector(file_rhs), dtype=torch.float32).reshape(1, 64, 64)
    flags = torch.tensor(read_flags(file_flags), dtype=torch.float32).reshape(1, 64, 64)
    sol = torch.tensor(load_vector(file_sol), dtype=torch.float32).reshape(64, 64)

    model = FluidNet()
    x = torch.cat([rhs, flags]).unsqueeze(0)
    y = model(x)

    print(y.shape)
    summary(model, (2, 64, 64))
