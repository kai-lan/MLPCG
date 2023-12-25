import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
import math

from torch.utils.cpp_extension import load
from lib.GLOBAL_VARS import *
from lib.global_clock import *

smblock3d = load(name='smblock3d',
                 sources=[f'{SOURCE_PATH}/torch_extension/sm_block_3d.cpp', f'{SOURCE_PATH}/torch_extension/sm_block_3d_kernel.cu'])
smblocktrans3d = load(name='smblocktrans3d',
                 sources=[f'{SOURCE_PATH}/torch_extension/sm_block_trans_3d.cpp', f'{SOURCE_PATH}/torch_extension/sm_block_trans_3d_kernel.cu'])
smlinear3d = load(name='smlinear3d',
                  sources=[f'{SOURCE_PATH}/torch_extension/sm_linear_3d.cpp', f'{SOURCE_PATH}/torch_extension/sm_linear_3d_kernel.cu'])

######################
# SM block
######################
class SMBlockFunction3D(torch.autograd.Function):
    @staticmethod
    def inference(image, x, weights, bias, timer=None):
        if timer: timer.start('Padding')
        image = F.pad(image, (1,)*6)
        x = F.pad(x, (1,)*6)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Padding')

        if timer: timer.start('Forward')
        y, = smblock3d.inference(image, x, weights, bias)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        return y
    @staticmethod
    def forward(ctx, image, x, weights, bias):
        image = F.pad(image, (1,)*6)
        x = F.pad(x, (1,)*6)
        y, = smblock3d.forward(image, x, weights, bias)
        ctx.save_for_backward(image, x, weights, bias)
        return y
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, x, weights, bias = ctx.saved_tensors
        grad_x, grad_w, grad_b, = smblock3d.backward(grad_output.contiguous(), image, x, weights, bias)
        return None, grad_x, grad_w, grad_b

class SMBlockTransFunction3D(torch.autograd.Function):
    @staticmethod
    def inference(image, x, weights, bias, timer=None):
        if timer: timer.start('Padding')
        image = F.pad(image, (1,)*6) # 3, N+2, N+2
        x = F.pad(x, (1,)*6) # bs, 1, N+2, N+2
        if timer:
            torch.cuda.synchronize()
            timer.stop('Padding')
        if timer: timer.start('Forward')
        y, = smblock3d.inference(image, x, weights, bias)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        return y

class SmallSMBlock3D(BaseModel):
    def __init__(self, num_imgs, mask=False):
        super().__init__()
        self.mask = mask
        self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        self.bias = nn.Parameter(torch.ones(27))
        self.reset_parameters(self.weight, self.bias)
    def forward(self, image, x):
        y = SMBlockFunction3D.apply(image, x, self.weight, self.bias)
        if self.mask:
            y = torch.where(image[1] == 0, 0.0, y) # mask out non-fluid
        return y
    def eval_forward(self, image, x, timer=None):
        y = SMBlockFunction3D.inference(image, x, self.weight, self.bias, timer)
        if self.mask:
            y = torch.where(image[1] == 0, 0.0, y)
        return y

class SmallSMBlockTrans3D(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        self.bias = nn.Parameter(torch.ones(27))
        self.reset_parameters(self.weight, self.bias)
    def eval_forward(self, image, x, timer=None):
        return SMBlockTransFunction3D.inference(image, x, self.weight, self.bias, timer)

class SmallSMBlock3DPY(BaseModel):
    def __init__(self, num_imgs):
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


######################
# SM linear
######################
class SMLinearFunction3D(torch.autograd.Function):
    @staticmethod
    def inference(image, weights, bias, timer=None):
        if timer: timer.start('Forward')
        z, y, = smlinear3d.inference(image, weights, bias)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        return z
    @staticmethod
    def forward(ctx, image, weights, bias, timer=None):
        if timer: timer.start('Forward')
        z, y, = smlinear3d.forward(image, weights, bias)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        if timer: timer.start('Saving for backward')
        ctx.save_for_backward(y)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Saving for backward')
        return z
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        y, = ctx.saved_tensors
        grad_w, grad_b, = smlinear3d.backward(grad_output, y)
        return None, grad_w, grad_b

class SmallLinearBlock3D(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        self.bias = nn.Parameter(torch.ones(27))
        self.reset_parameters(self.weight, self.bias)
    def forward(self, image):
        return SMLinearFunction3D.apply(image, self.weight, self.bias)
    def eval_forward(self, image, timer=None):
        return SMLinearFunction3D.inference(image, self.weight, self.bias, timer)

class SmallLinearBlock3DNew(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = torch.ones(27, num_imgs, 3, 3, 3)
        self.bias = torch.ones(27)
        self.reset_parameters(self.weight, self.bias)
        self.weight = nn.Parameter(self.weight.mean(dim=0, keepdim=True))
        self.bias = nn.Parameter(self.bias.mean(dim=0, keepdim=True))
    def forward(self, image):
        return SMLinearFunction3D.apply(image, self.weight, self.bias)
    def eval_forward(self, image, timer=None):
        return SMLinearFunction3D.inference(image, self.weight, self.bias, timer)

class SmallLinearBlock3DPY(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv3d(num_imgs, 27, kernel_size=3, padding='same')
        self.reset_parameters(self.KL.weight, self.KL.bias)
    def forward(self, image):
        K = self.KL(image) # num_imgs x N x N x N -> 27 x N x N x N
        return K.mean()


######################
# Full SM Model
######################
class SmallSMModelDn3D(BaseModel):
    def __init__(self, n, num_imgs, interpolation_mode='trilinear', mask=False, swap_sm_order=False):
        super().__init__()
        self.n = n
        self.mode = interpolation_mode
        self.swap_ord = swap_sm_order
        self.pre = nn.ModuleList([SmallSMBlock3D(num_imgs) for _ in range(n-1)])
        self.pre.insert(0, SmallSMBlock3D(num_imgs, mask))
        self.post = nn.ModuleList([SmallSMBlock3D(num_imgs) for _ in range(n-1)])
        self.post.insert(0, SmallSMBlock3D(num_imgs, mask))

        self.l = SmallSMBlock3D(num_imgs)

        self.c0 = nn.ModuleList([SmallLinearBlock3DNew(num_imgs) for _ in range(n)])
        self.c1 = nn.ModuleList([SmallLinearBlock3DNew(num_imgs) for _ in range(n)])

    def eval_forward(self, image, b, timer, c0=[], c1=[]):
        if c0:
            c0_cached = True
        else:
            c0_cached = False
            c0.extend([None for _ in range(self.n)])
        if c1:
            c1_cached = True
        else:
            c1_cached = False
            c1.extend([None for _ in range(self.n)])

        timer.start('SM block')
        x = [self.pre[0].eval_forward(image, b, timer)]
        imgs = [image]
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(1, self.n):
            timer.start('Downsampling')
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
            torch.cuda.synchronize()
            timer.stop('Downsampling')

            timer.start('SM block')
            x[-1] = self.pre[i].eval_forward(imgs[-1], x[-1], timer)
            torch.cuda.synchronize()
            timer.stop('SM block')

        timer.start('Downsampling')
        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
        torch.cuda.synchronize()
        timer.stop('Downsampling')

        timer.start('SM block')
        x[-1] = self.l.eval_forward(imgs[-1], x[-1], timer)
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(self.n, 0, -1):
            timer.start('Upsamping')
            x[i] = F.interpolate(x[i], scale_factor=2, mode=self.mode)
            torch.cuda.synchronize()
            timer.stop('Upsamping')

            timer.start('SM linear')
            if not c0_cached:
                c0[i-1] = self.c0[i-1].eval_forward(imgs[i-1], timer)
            if not c1_cached:
                c1[i-1] = self.c1[i-1].eval_forward(imgs[i-1], timer)
            torch.cuda.synchronize()
            timer.stop('SM linear')

            if self.swap_ord:
                timer.start('Linear combination')
                x[i-1] = c0[i-1] * x[i-1] + c1[i-1] * x[i]
                torch.cuda.synchronize()
                timer.stop('Linear combination')
                timer.start('SM block')
                x[i-1] = self.post[i-1].eval_forward(imgs[i-1], x[i-1], timer)
                torch.cuda.synchronize()
                timer.stop('SM block')
            else:
                timer.start('SM block')
                x[i] = self.post[i-1].eval_forward(imgs[i-1], x[i], timer)
                torch.cuda.synchronize()
                timer.stop('SM block')
                timer.start('Linear combination')
                x[i-1] = c0[i-1] * x[i-1] + c1[i-1] * x[i]
                torch.cuda.synchronize()
                timer.stop('Linear combination')

        return x[0]

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
            x[i] = F.interpolate(x[i], scale_factor=2, mode=self.mode)
            c0 = self.c0[i-1](imgs[i-1])
            c1 = self.c1[i-1](imgs[i-1])
            if self.swap_ord:
                x[i-1] = c0 * x[i-1] + c1 * x[i]
                x[i-1] = self.post[i-1](imgs[i-1], x[i-1])
            else:
                x[i] = self.post[i-1](imgs[i-1], x[i])
                x[i-1] = c0 * x[i-1] + c1 * x[i]
        return x[0]

class SmallSMModelDn3DPY(BaseModel):
    def __init__(self, n, num_imgs):
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

if __name__ == '__main__':
    import os, sys, time
    path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(path + "/lib")
    from lib.read_data import *
    import matplotlib.pyplot as plt
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False # for debugging
    # torch.use_deterministic_algorithms(True)
    # torch.set_grad_enabled(False)

    num_imgs = 3
    cuda_device = torch.device("cuda")


    x = torch.rand(1, 1, 64, 64, 64, device=cuda_device)
    y = torch.rand(1, 1, 64, 64, 64, device=cuda_device)
    img = torch.rand(3, 64, 64, 64, device=cuda_device)
    model = SmallSMBlock3D(num_imgs).to(cuda_device)
    model1 = SmallSMBlock3DPY(num_imgs).to(cuda_device)


    z = model.forward(img, x)
    z1 = model1.forward(img, x)

    # # print((torch.bmm(x.flatten(2), z1.flatten(1).unsqueeze(-1)) - torch.bmm(y.flatten(2), z.flatten(1).unsqueeze(-1))).norm())
    print((z - z1).norm())
    # print(model.weight.norm(), model1.weight.norm())