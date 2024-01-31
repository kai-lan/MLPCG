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
        y, = smblocktrans3d.inference(image, x, weights, bias)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        return y
    @staticmethod
    def forward(ctx, image, x, weights, bias):
        image = F.pad(image, (1,)*6)
        x = F.pad(x, (1,)*6)
        y, = smblocktrans3d.forward(image, x, weights, bias)
        ctx.save_for_backward(image, x, weights, bias)
        return y
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, x, weights, bias = ctx.saved_tensors
        grad_x, grad_w, grad_b, = smblocktrans3d.backward(grad_output.contiguous(), image, x, weights, bias)
        return None, grad_x, grad_w, grad_b

class SmallSMBlock3D(BaseModel):
    def __init__(self, num_imgs=3, mask=False):
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
    def __init__(self, num_imgs=3, weight=None, bias=None):
        super().__init__()
        if weight is None:
            self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        else:
            self.weight = weight
        if bias is None:
            self.bias = nn.Parameter(torch.ones(27))
        else:
            self.bias = bias
        if weight is None and bias is None:
            self.reset_parameters(self.weight, self.bias)
    def forward(self, image, x):
        return SMBlockTransFunction3D.apply(image, x, self.weight, self.bias)
    def eval_forward(self, image, x, timer=None):
        return SMBlockTransFunction3D.inference(image, x, self.weight, self.bias, timer)


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

# class SmallLinearBlock3D(BaseModel):
#     def __init__(self, num_imgs=3):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
#         self.bias = nn.Parameter(torch.ones(27))
#         self.reset_parameters(self.weight, self.bias)
#     def forward(self, image):
#         return SMLinearFunction3D.apply(image, self.weight, self.bias)
#     def eval_forward(self, image, timer=None):
#         return SMLinearFunction3D.inference(image, self.weight, self.bias, timer)

class SmallLinearBlock3DNew(BaseModel):
    def __init__(self, num_imgs=3):
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



######################
# Full SM Model
######################
class SmallSMModelDn3D(BaseModel):
    def __init__(self, n, num_imgs=3, interpolation_mode='trilinear', mask=False, swap_sm_order=False):
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

    def eval_forward(self, image, b, timer, imgs=[], c0=[], c1=[]):
        if imgs:
            imgs_cached = True
        else:
            imgs_cached = False
            imgs.extend([None for _ in range(self.n)])
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
        if not imgs_cached:
            imgs[0] = image
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(1, self.n):
            timer.start('Downsampling')
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            if not imgs_cached:
                imgs[i] = F.avg_pool3d(imgs[i-1], (2, 2, 2))
            torch.cuda.synchronize()
            timer.stop('Downsampling')

            timer.start('SM block')
            x[-1] = self.pre[i].eval_forward(imgs[i], x[-1], timer)
            torch.cuda.synchronize()
            timer.stop('SM block')

        timer.start('Downsampling')
        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        fine_img = F.avg_pool3d(imgs[-1], (2, 2, 2))
        torch.cuda.synchronize()
        timer.stop('Downsampling')

        timer.start('SM block')
        x[-1] = self.l.eval_forward(fine_img, x[-1], timer)
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


######################
# SPD SM Model
######################
class SPDSMModelDn3D(BaseModel):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.l0 = nn.ModuleList()
        self.l1 = nn.ModuleList()
        self.l0_t = nn.ModuleList()
        self.l1_t = nn.ModuleList()
        for _ in range(n):
            self.l0.append(SmallSMBlock3D())
            self.l1.append(SmallSMBlock3D())
            self.l0_t.append(SmallSMBlockTrans3D(weight=self.l0[-1].weight, bias=self.l0[-1].bias))
            self.l1_t.append(SmallSMBlockTrans3D(weight=self.l1[-1].weight, bias=self.l1[-1].bias))

        self.l = SmallSMBlock3D()
        self.l_t = SmallSMBlockTrans3D(weight=self.l.weight, bias=self.l.bias)

        self.c0 = nn.ModuleList([SmallLinearBlock3DNew() for _ in range(n)])
        self.c1 = nn.ModuleList([SmallLinearBlock3DNew() for _ in range(n)])

    def eval_forward(self, image, b, timer=None, imgs=[], c0=[], c1=[]):
        x = [self.l0[0].eval_forward(image, b)]
        imgs = [image]
        c0, c1 = [], []
        for i in range(1, self.n):
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
            x[-1] = self.l0[i].eval_forward(imgs[-1], x[-1])

        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
        x[-1] = self.l.eval_forward(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2) / 8
            c0.insert(0, self.c0[i-1].eval_forward(imgs[i-1]))
            c1.insert(0, self.c1[i-1].eval_forward(imgs[i-1]))
            x[i] = self.l1[i-1].eval_forward(imgs[i-1], x[i])
            x[i-1] = c0[0] * x[i-1] + c1[0] * x[i]


        b = x[0]
        x = [b]
        for i in range(self.n):
            x.append(self.l1_t[i].eval_forward(imgs[i], x[i]))
            x[-1] = F.avg_pool3d(x[-1], (2, 2, 2))

        x[-1] = self.l_t.eval_forward(imgs[-1], x[-1])

        for i in range(self.n-1, -1, -1):
            x[i+1] = F.interpolate(x[i+1], scale_factor=2) / 8
            x[i] = c0[i] * x[i] + c1[i] * x[i+1]
            x[i] = self.l0_t[i].eval_forward(imgs[i], x[i])

        return x[0]

    def forward(self, image, b):
        x = [self.l0[0](image, b)]
        imgs = [image]
        c0, c1 = [], []
        for i in range(1, self.n):
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
            x[-1] = self.l0[i](imgs[-1], x[-1])

        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2) / 8
            c0.insert(0, self.c0[i-1](imgs[i-1]))
            c1.insert(0, self.c1[i-1](imgs[i-1]))
            x[i] = self.l1[i-1](imgs[i-1], x[i])
            x[i-1] = c0[0] * x[i-1] + c1[0] * x[i]


        b = x[0]
        x = [b]
        for i in range(self.n):
            x.append(self.l1_t[i](imgs[i], x[i]))
            x[-1] = F.avg_pool3d(x[-1], (2, 2, 2))

        x[-1] = self.l_t(imgs[-1], x[-1])

        for i in range(self.n-1, -1, -1):
            x[i+1] = F.interpolate(x[i+1], scale_factor=2) / 8
            x[i] = c0[i] * x[i] + c1[i] * x[i+1]
            x[i] = self.l0_t[i](imgs[i], x[i])

        return x[0]

####################
# Reduced SPD model
####################
class SmallSPDSMModelDn3D(BaseModel):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.l0 = nn.ModuleList()
        self.l0_t = nn.ModuleList()
        for _ in range(n):
            self.l0.append(SmallSMBlock3D())
            self.l0_t.append(SmallSMBlockTrans3D(weight=self.l0[-1].weight, bias=self.l0[-1].bias))

        self.l = SmallSMBlock3D()
        self.l_t = SmallSMBlockTrans3D(weight=self.l.weight, bias=self.l.bias)

        self.c0 = nn.ModuleList([SmallLinearBlock3DNew() for _ in range(n)])
        self.c1 = nn.ModuleList([SmallLinearBlock3DNew() for _ in range(n)])

    def forward(self, image, b):
        x = [self.l0[0](image, b)]
        imgs = [image]
        c0, c1 = [], []
        for i in range(1, self.n):
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
            x[-1] = self.l0[i](imgs[-1], x[-1])
        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])
        x[-1] = self.l_t(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2) / 8
            c0.insert(0, self.c0[i-1](imgs[i-1]))
            c1.insert(0, self.c1[i-1](imgs[i-1]))
            x[i-1] = c0[0]**2 * x[i-1] + c1[0]**2 * x[i]
            # x[i-1] = x[i-1] + x[i]
            x[i-1] = self.l0_t[i-1](imgs[i-1], x[i-1])
        return x[0]

    def eval_forward(self, image, b, timer=None, imgs=[], c0=[], c1=[]):
        x = [self.l0[0].eval_forward(image, b)]
        imgs = [image]
        c0, c1 = [], []
        for i in range(1, self.n):
            x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
            imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
            x[-1] = self.l0[i].eval_forward(imgs[-1], x[-1])
        x.append(F.avg_pool3d(x[-1], (2, 2, 2)))
        imgs.append(F.avg_pool3d(imgs[-1], (2, 2, 2)))
        x[-1] = self.l.eval_forward(imgs[-1], x[-1])
        x[-1] = self.l_t.eval_forward(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2) / 8
            c0.insert(0, self.c0[i-1].eval_forward(imgs[i-1]))
            c1.insert(0, self.c1[i-1].eval_forward(imgs[i-1]))
            x[i-1] = c0[0]**2 * x[i-1] + c1[0]**2 * x[i]

            # x[i-1] = x[i-1] + x[i]
            x[i-1] = self.l0_t[i-1].eval_forward(imgs[i-1], x[i-1])
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

    # model = SPDSMModelDn3D(1).to(cuda_device)
    # for _ in range(100):
    x = torch.rand(128, 1, 128, 128, 128, device=cuda_device)
    y = torch.rand(128, 1, 128, 128, 128, device=cuda_device)
    img = torch.rand(3, 128, 128, 128, device=cuda_device)

    # model1 = SmallSMBlock3D().to(cuda_device)
    # model = SmallSMBlockTrans3D(weight=model1.weight, bias=model1.bias).to(cuda_device)
    # model.weight = model1.weight
    # model.bias = model1.bias
    model = SmallSPDSMModelDn3D(4).to(cuda_device)

    z = model(img, x)
    # z.sum().backward()

    z1 = model(img, y)
    a = torch.bmm(x.flatten(2), z1.flatten(1).unsqueeze(-1))
    b = torch.bmm(y.flatten(2), z.flatten(1).unsqueeze(-1))
    abs_diff = (a - b).norm().item()
    rel_diff = abs_diff / torch.sqrt(a.abs() * b.abs()).norm().item()

    print(abs_diff, rel_diff)
    # assert c < 1e-12
    exit()
    # print(model.bias)
    # z = model1(img, model(img, x))
    # z1 = model1(img, model(img, y))

    # optimizer = torch.optim.Adam(model1.parameters())

    # z.sum().backward()
    # optimizer.step()

    # print(model.bias)
    # print(model1.bias)


    # z = model.eval_forward(img, x)

    # print((z - z1).norm())
    # print(model.weight.norm(), model1.weight.norm())

    for _ in range(2):
        y = model(img, x)
        y1 = model1(img, x)
        y.sum().backward()
        y1.sum().backward()
    torch.cuda.synchronize()

    # timer = GlobalClock()

    iters = 10
    forward = 0.0
    backward = 0.0
    for _ in range(iters):
        start = time.perf_counter()
        y = model(img, x)
        torch.cuda.synchronize()
        forward += time.perf_counter() - start

        start = time.perf_counter()
        y.sum().backward()
        torch.cuda.synchronize()
        backward += time.perf_counter() - start

    print('PyTorch\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))

    forward = 0.0
    backward = 0.0
    for _ in range(iters):
        start = time.perf_counter()
        y1 = model1(img, x)
        torch.cuda.synchronize()
        forward += time.perf_counter() - start

        start = time.perf_counter()
        y1.sum().backward()
        torch.cuda.synchronize()
        backward += time.perf_counter() - start

    print('CUDA kernel\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))
