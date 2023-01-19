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

from lib.cell_type import flagsToOccupancy
from lib.velocity_update import velocityUpdate
# from lib import FlexiUNet, MultiScaleNet, UNet, UNet3D, fluid
from flexiunet import FlexiUNet

class _ScaleNet(nn.Module):
    # normalizeInputThreshold : don't normalize input noise
    def __init__(self, normalizeInputThreshold=0.00001):
        super(_ScaleNet, self).__init__()
        self.normalizeInputThreshold = normalizeInputThreshold
    def forward(self, x):
        bsz = x.size(0)
        # Rehaspe form (b x chan x d x h x w) to (b x -1)
        y = x.view(bsz, -1)
        # Calculate std using Bessel's correction (correction with n/n-1)
        std = torch.std(y, dim=1, keepdim=True)  # output is size (b x 1)
        scale = torch.clamp(std, self.normalizeInputThreshold, inf)
        scale = scale.view(bsz, 1, 1, 1, 1)

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
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='replicate')
        self.conv5 = nn.Conv2d(8, 1, kernel_size=3, )

    def forward(self, x, it, folder):
        # Input: [bs, 2, 1, Ny, Nx]
        div_U = x[:, 0] # Pointer to x, if in-place operations
        flags = x[:, 1] # Pointer to x, if in-place operations
        # Normalize input
        s = self.scale(div_U)
        div_U /= s

        # FlagsToOccupancy creates a [0,1] grid out of the manta flags
        flags[:] = flagsToOccupancy(flags).squeeze(1)

        if not self.is3D:
            # Squeeze unary dimension as we are in 2D: Ok, but we have to use Conv2D
            x = torch.squeeze(x, 2)

        # Inital layers
        x = F.relu(self.conv1(x))

        # We divide the network in 3 banks, applying average pooling
        x1 = self.modDown1(x)
        x2 = self.modDown2(x)

        # Process every bank in parallel
        x0 = self.convBank(x)
        x1 = self.convBank(x1)
        x2 = self.convBank(x2)

        # Upsample banks 1 and 2 to bank 0 size and accumulate inputs

        #x1 = self.upscale1(x1)
        #x2 = self.upscale2(x2)

        x1 = F.interpolate(x1, scale_factor=2)
        x2 = F.interpolate(x2, scale_factor=4)
        #x1 = self.deconv1(x1)
        #x2 = self.deconv2(x2)

        #x = torch.cat((x0, x1, x2), dim=1)
        x = x0 + x1 + x2

        # Apply last 2 convolutions
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Output pressure (1 chan)
        p = self.convOut(x)

        # Add back the unary dimension
        if not self.is3D:
            p = torch.unsqueeze(p, 2)

        # Correct U = UDiv - grad(p)
        # flags is the one with Manta's values, not occupancy in [0,1]

        velocityUpdate(pressure=p, U=UDiv, flags=flags)

        # We now UNDO the scale factor we applied on the input.
        if self.mconf['normalizeInput']:
            p = torch.mul(p, s)  # Applies p' = *= scale
            UDiv = torch.mul(UDiv, s)

        # Set BCs after velocity update.
        #UDiv = fluid.setWallBcs(UDiv, flags)

        return p, UDiv, time
