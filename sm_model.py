import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
import math

from torch.utils.cpp_extension import load
from lib.GLOBAL_VARS import *
from lib.global_clock import *

smblock = load(name='smblock',
               sources=[f'{SOURCE_PATH}/torch_extension/sm_block.cpp',
               f'{SOURCE_PATH}/torch_extension/sm_block_kernel.cu'],
               extra_include_paths=[f'{SOURCE_PATH}/torch_extension'])
smlinear = load(name='smlinear',
                sources=[f'{SOURCE_PATH}/torch_extension/sm_linear.cpp', f'{SOURCE_PATH}/torch_extension/sm_linear_kernel.cu'])
smblock3d = load(name='smblock3d',
                 sources=[f'{SOURCE_PATH}/torch_extension/sm_block_3d.cpp', f'{SOURCE_PATH}/torch_extension/sm_block_3d_kernel.cu'])
smlinear3d = load(name='smlinear3d',
                  sources=[f'{SOURCE_PATH}/torch_extension/sm_linear_3d.cpp', f'{SOURCE_PATH}/torch_extension/sm_linear_3d_kernel.cu'])

######################
# SM block 2D/3D
######################
class SMBlockFunction(torch.autograd.Function):
    @staticmethod
    def inference(image, x, weights, bias, timer=None):
        if timer: timer.start('Padding')
        image = F.pad(image, (1,)*4) # 3, N+2, N+2
        x = F.pad(x, (1,)*4) # bs, 1, N+2, N+2
        if timer:
            torch.cuda.synchronize()
            timer.stop('Padding')
        if timer: timer.start('Forward')
        y, = smblock.forward(image, x, weights, bias) # TODO implement inference
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        return y
    @staticmethod
    def forward(ctx, image, x, weights, bias):
        image = F.pad(image, (1,)*4) # 3, N+2, N+2
        x = F.pad(x, (1,)*4) # bs, 1, N+2, N+2
        ctx.save_for_backward(image, x, weights, bias)
        y, = smblock.forward(image, x, weights, bias)
        return y
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, x, weights, bias = ctx.saved_tensors
        grad_x, grad_w, grad_b, = smblock.backward(grad_output.contiguous(), image, x, weights, bias)
        return None, grad_x, grad_w, grad_b

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

class SmallSMBlock(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(9, num_imgs, 3, 3))
        self.bias = nn.Parameter(torch.ones(9))
        self.reset_parameters(self.weight, self.bias)
    def forward(self, image, x):
        return SMBlockFunction.apply(image, x, self.weight, self.bias)
    def eval_forward(self, image, x, timer=None):
        return SMBlockFunction.inference(image, x, self.weight, self.bias, timer)


class SmallSMBlockPY(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv2d(num_imgs, 9, kernel_size=3, padding='same', bias=True)
        self.reset_parameters(self.KL.weight, self.KL.bias)
    def forward(self, image, x): # num_imgs x N x N, bs x 1 x N x N
        K = self.KL(image) # num_imgs x N x N -> 9 x N x N
        K = K.permute((1, 2, 0)) # 9 x N x N -> N x N x 9
        K = K.unflatten(2, (3, 3)) # N x N x 9 -> N x N x 3 x 3
        x = F.pad(x, (1, 1, 1, 1)) # bs x 1 x N x N -> bs x 1 x (N+2) x (N+2)
        x = x.unfold(2, 3, 1).unfold(3, 3, 1) # bs x 1 x (N+2) x (N+2) -> bs x 1 x N x N x 3 x 3
        y = (x * K).sum(dim=(-2, -1))
        return y

class SmallSMBlock3D(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        self.bias = nn.Parameter(torch.ones(27))
        self.reset_parameters(self.weight, self.bias)
    def forward(self, image, x):
        y = SMBlockFunction3D.apply(image, x, self.weight, self.bias)
        return y
    def eval_forward(self, image, x, timer):
        y = SMBlockFunction3D.inference(image, x, self.weight, self.bias, timer)
        return y

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
# SM linear 2D/3D
######################
class SMLinearFunction(torch.autograd.Function):
    @staticmethod
    def inference(image, weights, bias, timer=None):
        if timer: timer.start('Padding')
        # image = F.pad(image, (1,)*4)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Padding')

        if timer: timer.start('Forward')
        z, = smlinear.inference(image, weights, bias)
        if timer:
            torch.cuda.synchronize()
            timer.stop('Forward')
        return z
    @staticmethod
    def forward(ctx, image, weights, bias):
        # image = F.pad(image, (1,)*4)
        z, y, = smlinear.forward(image, weights, bias)
        ctx.save_for_backward(y)
        return z
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        y, = ctx.saved_tensors
        grad_w, grad_b, = smlinear.backward(grad_output.contiguous(), y)
        return None, grad_w, grad_b

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

class SmallLinearBlock(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(9, num_imgs, 3, 3))
        self.bias = nn.Parameter(torch.ones(9))
        self.reset_parameters(self.weight, self.bias)
    def forward(self, image):
        return SMLinearFunction.apply(image, self.weight, self.bias)
    def eval_forward(self, image, timer=None):
        return SMLinearFunction.inference(image, self.weight, self.bias, timer)

class SmallLinearBlockPY(BaseModel):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv2d(num_imgs, 9, kernel_size=3, padding='same')
        self.reset_parameters(self.KL.weight, self.KL.bias)
    def forward(self, image):
        K = self.KL(image) # num_imgs x N x N -> 9 x N x N
        return K.mean()

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
# Full SM Model 2D/3D
######################
# General SmallSM model for any number of coarsening levels
class SmallSMModelDn(BaseModel):
    def __init__(self, n, num_imgs):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList([SmallSMBlock(num_imgs) for _ in range(n)])
        self.post = nn.ModuleList([SmallSMBlock(num_imgs) for _ in range(n)])

        self.l = SmallSMBlock(num_imgs)

        self.c0 = nn.ModuleList([SmallLinearBlock(num_imgs) for _ in range(n)])
        self.c1 = nn.ModuleList([SmallLinearBlock(num_imgs) for _ in range(n)])

    def eval_forward(self, image, b, timer):
        timer.start('SM block')
        x = [self.pre[0].eval_forward(image, b, timer)]
        imgs = [image]
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(1, self.n):
            timer.start('Downsampling')
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
            torch.cuda.synchronize()
            timer.stop('Downsampling')
            timer.start('SM block')
            x[-1] = self.pre[i].eval_forward(imgs[-1], x[-1], timer)
            torch.cuda.synchronize()
            timer.stop('SM block')

        timer.start('Downsampling')
        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
        torch.cuda.synchronize()
        timer.stop('Downsampling')

        timer.start('SM block')
        x[-1] = self.l.eval_forward(imgs[-1], x[-1], timer)
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(self.n, 0, -1):
            timer.start('Upsamping')
            x[i] = F.interpolate(x[i], scale_factor=2)
            torch.cuda.synchronize()
            timer.stop('Upsamping')

            timer.start('SM block')
            x[i] = self.post[i-1].eval_forward(imgs[i-1], x[i], timer)
            torch.cuda.synchronize()
            timer.stop('SM block')

            timer.start('SM linear')
            c0 = self.c0[i-1].eval_forward(imgs[i-1], timer)
            c1 = self.c1[i-1].eval_forward(imgs[i-1], timer)
            torch.cuda.synchronize()
            timer.stop('SM linear')

            timer.start('Linear combination')
            x[i-1] = c0 * x[i-1] + c1 * x[i]
            torch.cuda.synchronize()
            timer.stop('Linear combination')

        return x[0]

    def forward(self, image, b):
        x = [self.pre[0](image, b)]
        imgs = [image]

        for i in range(1, self.n):
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
            x[-1] = self.pre[i](imgs[-1], x[-1])

        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = self.post[i-1](imgs[i-1], x[i])
            c0 = self.c0[i-1](imgs[i-1])
            c1 = self.c1[i-1](imgs[i-1])
            x[i-1] = c0 * x[i-1] + c1 * x[i]

        return x[0]

class SmallSMModelDnPY(BaseModel):
    def __init__(self, n, num_imgs):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList([SmallSMBlockPY(num_imgs) for _ in range(n)])
        self.post = nn.ModuleList([SmallSMBlockPY(num_imgs) for _ in range(n)])

        self.l = SmallSMBlockPY(num_imgs)

        self.c0 = nn.ModuleList([SmallLinearBlockPY(num_imgs) for _ in range(n)])
        self.c1 = nn.ModuleList([SmallLinearBlockPY(num_imgs) for _ in range(n)])

    def eval_forward(self, image, b, timer):
        timer.start('SM block')
        x = [self.pre[0].eval_forward(image, b)]
        imgs = [image]
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(1, self.n):
            timer.start('Downsampling')
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
            torch.cuda.synchronize()
            timer.stop('Downsampling')
            timer.start('SM block')
            x[-1] = self.pre[i].eval_forward(imgs[-1], x[-1])
            torch.cuda.synchronize()
            timer.stop('SM block')

        timer.start('Downsampling')
        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
        torch.cuda.synchronize()
        timer.stop('Downsampling')

        timer.start('SM block')
        x[-1] = self.l.eval_forward(imgs[-1], x[-1])
        torch.cuda.synchronize()
        timer.stop('SM block')

        for i in range(self.n, 0, -1):
            timer.start('Upsamping')
            x[i] = F.interpolate(x[i], scale_factor=2)
            torch.cuda.synchronize()
            timer.stop('Upsamping')

            timer.start('SM block')
            x[i] = self.post[i-1].eval_forward(imgs[i-1], x[i])
            torch.cuda.synchronize()
            timer.stop('SM block')

            timer.start('SM linear')
            c0 = self.c0[i-1].eval_forward(imgs[i-1])
            c1 = self.c1[i-1].eval_forward(imgs[i-1])
            torch.cuda.synchronize()
            timer.stop('SM linear')

            timer.start('Linear combination')
            x[i-1] = c0 * x[i-1] + c1 * x[i]
            torch.cuda.synchronize()
            timer.stop('Linear combination')

        return x[0]

    def forward(self, image, b):
        x = [self.pre[0](image, b)]
        imgs = [image]

        for i in range(1, self.n):
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
            x[-1] = self.pre[i](imgs[-1], x[-1])

        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = self.post[i-1](imgs[i-1], x[i])
            x[i-1] = self.c0[i-1](imgs[i-1]) * x[i-1] + self.c1[i-1](imgs[i-1]) * x[i]

        return x[0]


# 3D: General SmallSM model for any number of coarsening levels
class SmallSMModelDn3D(BaseModel):
    def __init__(self, n, num_imgs):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList([SmallSMBlock3D(num_imgs) for _ in range(n)])
        self.post = nn.ModuleList([SmallSMBlock3D(num_imgs) for _ in range(n)])

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
            x[i] = F.interpolate(x[i], scale_factor=2)
            torch.cuda.synchronize()
            timer.stop('Upsamping')

            timer.start('SM block')
            x[i] = self.post[i-1].eval_forward(imgs[i-1], x[i], timer)
            torch.cuda.synchronize()
            timer.stop('SM block')

            timer.start('SM linear')
            if not c0_cached:
                c0[i-1] = self.c0[i-1].eval_forward(imgs[i-1], timer)
            if not c1_cached:
                c1[i-1] = self.c1[i-1].eval_forward(imgs[i-1], timer)
            torch.cuda.synchronize()
            timer.stop('SM linear')

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
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = self.post[i-1](imgs[i-1], x[i])
            c0 = self.c0[i-1](imgs[i-1])
            c1 = self.c1[i-1](imgs[i-1])
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
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False # for debugging
    # torch.use_deterministic_algorithms(True)

    N = 128
    frame = 200
    num_imgs = 3
    cuda_device = torch.device("cuda")

    file_rhs = os.path.join(DATA_PATH, f"dambreak_N{N}_200_3D", f"div_v_star_{frame}.bin")
    file_flags = os.path.join(DATA_PATH, f"dambreak_N{N}_200_3D", f"flags_{frame}.bin")
    rhs = torch.tensor(load_vector(file_rhs), dtype=torch.get_default_dtype())
    image = read_flags(file_flags)
    fluid_cells = np.argwhere(image == FLUID).ravel()
    flags = torch.tensor(convert_to_binary_images(image, num_imgs), dtype=torch.get_default_dtype())

    # file_A = f"{DATA_PATH}/dambreak_N{N}_200_3D/A_{frame}.bin"
    # A = readA_sparse(file_A)
    # A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, A.shape, dtype=torch.float32, device=cuda_device)
    # A_csc = A.to_sparse_csc()

    # file_sol = f"{DATA_PATH}/dambreak_N{N}_200_3D/pressure_{frame}.bin"
    # sol = torch.tensor(load_vector(file_sol), dtype=torch.float32, device=cuda_device)


    # model = SmallSMBlockPY(num_imgs).to(cuda_device)
    # model1 = SmallSMBlock(num_imgs).to(cuda_device)
    # model = SmallLinearBlockPY(num_imgs).to(cuda_device)
    # model1 = SmallLinearBlock(num_imgs).to(cuda_device)

    # model = SmallLinearBlock3DPY(num_imgs).to(cuda_device)
    # model1 = SmallLinearBlock3D(num_imgs).to(cuda_device)
    model = SmallLinearBlock3DNew(num_imgs).to(cuda_device)
    model1 = SmallLinearBlock3D(num_imgs).to(cuda_device)
    # print(model1.weight.mean(dim=0), model1.bias.mean(dim=0))

    # model = SmallSMModelDnPY(6, num_imgs).to(cuda_device)
    # model1 = SmallSMModelDn(6, num_imgs).to(cuda_device)

    img_shape = (num_imgs, N, N, N)
    rhs_shape = (1, 1, N, N, N)
    image = flags.reshape(img_shape).to(cuda_device)

    image = torch.rand(img_shape).to(cuda_device)
    print(image.mean(), image.var())
    z = model(image)
    print(z)
    exit()
    # x = torch.zeros(np.prod(rhs_shape))
    # x[fluid_cells] = rhs
    x = rhs
    x = x.reshape(rhs_shape).expand((1,)+ rhs_shape[1:]).to(cuda_device)
    x1 = x.detach().clone()

    # torch.set_grad_enabled(False)
    # model1.eval()
    # print(model1.training)
    # timer = GlobalClock()

    # for _ in range(10):
    #     y = model(image)
    #     y1 = model1.eval_forward(image)
    #     # y.sum().backward()
    #     # y1.sum().backward()
    # torch.cuda.synchronize()

    # timer = GlobalClock()

    # iters = 100
    # forward = 0.0
    # backward = 0.0
    # for _ in range(iters):
    #     start = time.perf_counter()
    #     y = model(image)
    #     torch.cuda.synchronize()
    #     forward += time.perf_counter() - start

    #     start = time.perf_counter()
    #     # y.sum().backward()
    #     torch.cuda.synchronize()
    #     backward += time.perf_counter() - start

    # print('PyTorch\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))

    # # grad_output = torch.zeros(1).to(cuda_device)
    # # padded_image = F.pad(image, (1,)*6)
    # timer.report()
    # timer = GlobalClock()

    # forward = 0.0
    # backward = 0.0
    # for _ in range(iters):
    #     start = time.perf_counter()
    #     y1 = model1.eval_forward(image)
    #     torch.cuda.synchronize()
    #     forward += time.perf_counter() - start

    #     start = time.perf_counter()
    #     # y1.sum().backward()
    #     torch.cuda.synchronize()
    #     backward += time.perf_counter() - start

    # print('CUDA kernel\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))
    # timer.report()
    # print((y - y1).abs().max())
    # print((y - y1).abs().norm())
    # y.sum().backward()
    # y1.sum().backward()
    # print((model.KL.bias.grad - model1.bias.grad).abs().max())
    # print((model.KL.weight.grad - model1.weight.grad).abs().max())
    # print((model.pre[1].KL.weight.grad - model1.pre[1].weight.grad).abs().max())
    # print((x.grad - x1.grad).abs().max())


