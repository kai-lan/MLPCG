import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
import math

# import smblock3d # implemented in torch_extension/sm_block_kernel.cu
from torch.utils.cpp_extension import load
from lib.GLOBAL_VARS import *

smblock = load(name='smblock',
               sources=[f'{SOURCE_PATH}/torch_extension/sm_block.cpp', f'{SOURCE_PATH}/torch_extension/sm_block_kernel.cu'])
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
    def forward(ctx, image, x, weights, bias):
        image = F.pad(image, (1,)*4) # 1, N+2, N+2
        x = F.pad(x, (1,)*4) # bs, 1, N+2, N+2
        ctx.save_for_backward(image, x, weights, bias)
        y, = smblock.forward(image, x, weights, bias)
        return y
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, x, weights, bias = ctx.saved_tensors
        grad_x, grad_w, grad_b, = smblock.backward(grad_output.contiguous(), image, x, weights, bias)
        return None, grad_x, grad_w, grad_b

class SmallSMBlock(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(9, num_imgs, 3, 3))
        self.bias = nn.Parameter(torch.ones(9))
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, image, x):
        return SMBlockFunction.apply(image, x, self.weight, self.bias)

class SmallSMBlockPY(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv2d(num_imgs, 9, kernel_size=3, padding='same', bias=True)
        torch.manual_seed(0)
        nn.init.kaiming_uniform_(self.KL.weight, a=math.sqrt(5))
        if self.KL.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.KL.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.KL.bias, -bound, bound)
    def forward(self, image, x): # num_imgs x N x N, bs x 1 x N x N
        K = self.KL(image) # num_imgs x N x N -> 9 x N x N
        K = K.permute((1, 2, 0)) # 9 x N x N -> N x N x 9
        K = K.unflatten(2, (3, 3)) # N x N x 9 -> N x N x 3 x 3
        x = F.pad(x, (1, 1, 1, 1)) # bs x 1 x N x N -> bs x 1 x (N+2) x (N+2)
        x = x.unfold(2, 3, 1).unfold(3, 3, 1) # bs x 1 x (N+2) x (N+2) -> bs x 1 x N x N x 3 x 3
        y = (x * K).sum(dim=(-2, -1))
        return y

class SMBlockFunction3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, x, weights, bias):
        image = F.pad(image, (1,)*6)
        x = F.pad(x, (1,)*6)
        ctx.save_for_backward(image, x, weights, bias)
        y, = smblock3d.forward(image, x, weights, bias)
        return y
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, x, weights, bias = ctx.saved_tensors
        grad_x, grad_w, grad_b, = smblock3d.backward(grad_output.contiguous(), image, x, weights, bias)
        return None, grad_x, grad_w, grad_b

class SmallSMBlock3D(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        self.bias = nn.Parameter(torch.ones(27))
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, image, x):
        y = SMBlockFunction3D.apply(image, x, self.weight, self.bias)
        return y

class SmallSMBlock3DPY(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv3d(num_imgs, 27, kernel_size=3, padding='same', bias=True)
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.KL.weight, a=math.sqrt(5))
        if self.KL.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.KL.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.KL.bias, -bound, bound)
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
    def forward(ctx, image, weights, bias):
        image = F.pad(image, (1,)*4)
        ctx.save_for_backward(image)
        z, = smlinear.forward(image, weights, bias)
        return z
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, = ctx.saved_tensors
        grad_w, grad_b, = smlinear.backward(grad_output.contiguous(), image)
        return None, grad_w, grad_b

class SmallLinearBlock(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(9, num_imgs, 3, 3))
        self.bias = nn.Parameter(torch.ones(9))
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, image, *placeholder):
        return SMLinearFunction.apply(image, self.weight, self.bias)

class SmallLinearBlockPY(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv2d(num_imgs, 9, kernel_size=3, padding='same')
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.KL.weight, a=math.sqrt(5))
        if self.KL.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.KL.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.KL.bias, -bound, bound)
    def forward(self, image, *placeholder):
        K = self.KL(image) # num_imgs x N x N -> 9 x N x N
        return K.mean()

class SMLinearFunction3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, weights, bias):
        image = F.pad(image, (1,)*6)
        ctx.save_for_backward(image)
        z, = smlinear3d.forward(image, weights, bias)
        return z
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, = ctx.saved_tensors
        grad_w, grad_b, = smlinear3d.backward(grad_output.contiguous(), image)
        return None, grad_w, grad_b

class SmallLinearBlock3D(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(27, num_imgs, 3, 3, 3))
        self.bias = nn.Parameter(torch.ones(27))
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, image, *placeholder):
        return SMLinearFunction3D.apply(image, self.weight, self.bias)

class SmallLinearBlock3DPY(nn.Module):
    def __init__(self, num_imgs):
        super().__init__()
        self.KL = nn.Conv3d(num_imgs, 27, kernel_size=3, padding='same')
        torch.manual_seed(0)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.KL.weight, a=math.sqrt(5))
        if self.KL.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.KL.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.KL.bias, -bound, bound)
    def forward(self, image, *placeholder):
        K = self.KL(image) # num_imgs x N x N x N -> 27 x N x N x N
        return K.mean()


# General SmallSM model for any number of coarsening levels
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
        x = [self.pre[0](image, b)]
        # x = [F.elu(self.pre[0](image, b))]
        imgs = [image]

        for i in range(1, self.n):
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
            x[-1] = self.pre[i](imgs[-1], x[-1])
            # x[-1] = F.elu(self.pre[i](imgs[-1], x[-1]))

        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])
        # x[-1] = F.elu(self.l(imgs[-1], x[-1]))

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = self.post[i-1](imgs[i-1], x[i])
            # x[i] = F.elu(self.post[i-1](imgs[i-1], x[i]))
            x[i-1] = self.c0[i-1](imgs[i-1]) * x[i-1] + self.c1[i-1](imgs[i-1]) * x[i]

        return x[0]

class SmallSMModelDnPY(BaseModel):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList()
        self.post = nn.ModuleList()
        for _ in range(n):
            self.pre.append(SmallSMBlockPY())
            self.post.append(SmallSMBlockPY())
        self.l = SmallSMBlockPY()
        self.c0 = nn.ModuleList()
        self.c1 = nn.ModuleList()
        for _ in range(n):
            self.c0.append(SmallLinearBlock())
            self.c1.append(SmallLinearBlock())

    def forward(self, image, b):
        x = [self.pre[0](image, b)]
        # x = [F.elu(self.pre[0](image, b))]
        imgs = [image]

        for i in range(1, self.n):
            x.append(F.avg_pool2d(x[-1], (2, 2)))
            imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
            x[-1] = self.pre[i](imgs[-1], x[-1])
            # x[-1] = F.elu(self.pre[i](imgs[-1], x[-1]))

        x.append(F.avg_pool2d(x[-1], (2, 2)))
        imgs.append(F.avg_pool2d(imgs[-1], (2, 2)))
        x[-1] = self.l(imgs[-1], x[-1])
        # x[-1] = F.elu(self.l(imgs[-1], x[-1]))

        for i in range(self.n, 0, -1):
            x[i] = F.interpolate(x[i], scale_factor=2)
            x[i] = self.post[i-1](imgs[i-1], x[i])
            # x[i] = F.elu(self.post[i-1](imgs[i-1], x[i]))
            x[i-1] = self.c0[i-1](imgs[i-1]) * x[i-1] + self.c1[i-1](imgs[i-1]) * x[i]

        return x[0]


# 3D: General SmallSM model for any number of coarsening levels
class SmallSMModelDn3D(BaseModel):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList()
        self.post = nn.ModuleList()
        for _ in range(n):
            self.pre.append(SmallSMBlock3D())
            self.post.append(SmallSMBlock3D())
        self.l = SmallSMBlock3D()
        self.c0 = nn.ModuleList()
        self.c1 = nn.ModuleList()
        for _ in range(n):
            self.c0.append(SmallLinearBlock3D())
            self.c1.append(SmallLinearBlock3D())

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

class SmallSMModelDn3DPY(BaseModel):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList()
        self.post = nn.ModuleList()
        for _ in range(n):
            self.pre.append(SmallSMBlock3DPY())
            self.post.append(SmallSMBlock3DPY())
        self.l = SmallSMBlock3DPY()
        self.c0 = nn.ModuleList()
        self.c1 = nn.ModuleList()
        for _ in range(n):
            self.c0.append(SmallLinearBlock3DPY())
            self.c1.append(SmallLinearBlock3DPY())

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

# For multiple channels
class MultiSMBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.KL = nn.Conv2d(1, 9*nc, kernel_size=3, padding='same')
        self.nc = nc
    def forward(self, image, x): # 1 x N x N, bs x 1 x N x N
        K = self.KL(image) # 1 x N x N -> (2 x 9) x N x N
        K = K.unflatten(0, (self.nc, 9)) # (2 x 9) x N x N -> 2 x 9 x N x N
        K = K.permute((0, 2, 3, 1)) # 2 x 9 x N x N -> 2 x N x N x 9
        K = K.unflatten(-1, (3, 3)) # 2 x N x N x 9 -> 2 x N x N x 3 x 3
        x = F.pad(x, (1, 1, 1, 1)) # bs x 1 x N x N -> bs x 1 x (N+2) x (N+2)
        x = x.unfold(2, 3, 1).unfold(3, 3, 1) # bs x 1 x (N+2) x (N+2) -> bs x 1 x N x N x 3 x 3
        y = (x * K).sum(dim=(-2, -1)) # bs x 1 x N x N x 3 x 3, 2 x N x N x 3 x 3 -> bs x 2 x N x N
        return y


if __name__ == '__main__':
    import os, sys, time
    path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(path + "/lib")
    from lib.read_data import *
    import matplotlib.pyplot as plt
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True # for debugging
    # torch.use_deterministic_algorithms(True)

    N = 128
    frame = 100
    num_imgs = 3

    # file_A = os.path.join(path, "data_fluidnet", "dambreak_2D_64", f"A_{frame}.bin")
    file_rhs = os.path.join(DATA_PATH, f"dambreak_N{N}_200_3D", f"div_v_star_{frame}.bin")
    # file_sol = os.path.join(DATA_PATH, f"dambreak_N{N}_200_3D", f"pressure_{frame}.bin")
    file_flags = os.path.join(DATA_PATH, f"dambreak_N{N}_200_3D", f"flags_{frame}.bin")
    # A = readA_sparse(64, file_A, DIM=2)
    rhs = torch.tensor(load_vector(file_rhs), dtype=torch.float32)
    flags = torch.tensor(convert_to_binary_images(read_flags(file_flags), num_imgs), dtype=torch.float32)
    # sol = torch.tensor(load_vector(file_sol), dtype=torch.float32)
    # with h5py.File("test.hdf5", "r") as f:
    #   rhs = torch.tensor(f['rhs'], dtype=torch.float32)
    #   flags = torch.tensor(f['flags'], dtype=torch.float32)

    # torch.set_grad_enabled(False) # disable autograd globally

    # model = SmallSMBlock3DPY().cuda()
    # model1 = SmallSMBlock3D().cuda()
    model = SmallLinearBlock3DPY(num_imgs).cuda()
    model1 = SmallLinearBlock3D(num_imgs).cuda()
    # model = SmallSMBlockPY().cuda()
    # model1 = SmallSMBlock().cuda()
    # model = SmallSMModelDn3DPY(2).cuda()
    # model1 = SmallSMModelDn3D(2).cuda()

    img_shape = (num_imgs, N, N, N)
    rhs_shape = (1, 1, N, N, N)
    image = flags.reshape(img_shape).cuda()
    x = rhs.reshape(rhs_shape).expand((8,)+ img_shape).cuda()
    x.requires_grad = True
    x1 = rhs.reshape(rhs_shape).expand((8,)+ img_shape).cuda()
    x1.requires_grad = True

    # torch.set_grad_enabled(False)
    for _ in range(10):
        y = model(image, x)
        y1 = model1(image, x1)
        # y.sum().backward()
        # y1.sum().backward()

    torch.cuda.synchronize()

    iters = 100
    forward = 0.0
    backward = 0.0
    for _ in range(iters):
        start = time.time()
        y = model(image, x)
        torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        y.sum().backward()
        torch.cuda.synchronize()
        backward += time.time() - start

    print('PyTorch\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))

    forward = 0.0
    backward = 0.0
    for _ in range(iters):
        start = time.time()
        y1 = model1(image, x1)
        torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        y1.sum().backward()
        torch.cuda.synchronize()
        backward += time.time() - start

    print('CUDA kernel\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))

    # print((y - y1).abs().max())
    # y.sum().backward()
    # y1.sum().backward()
    # print((model.KL.bias.grad - model1.bias.grad).abs().mean())
    # print((model.KL.weight.grad - model1.weight.grad).abs().mean())
    # print((model.pre[1].KL.weight.grad - model1.pre[1].KL.weight.grad).abs().max())
    # print((x.grad - x1.grad).abs().max())


