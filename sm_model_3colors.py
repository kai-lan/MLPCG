import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
import math

from torch.utils.cpp_extension import load
from lib.GLOBAL_VARS import *


class SmallSMBlock3ColorsPY(nn.Module):
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
    def forward(self, image, x): # 2 x N x N, bs x 1 x N x N
        K = self.KL(image) # 2 x N x N -> 9 x N x N
        K = K.permute((1, 2, 0)) # 9 x N x N -> N x N x 9
        K = K.unflatten(2, (3, 3)) # N x N x 9 -> N x N x 3 x 3
        x = F.pad(x, (1, 1, 1, 1)) # bs x 1 x N x N -> bs x 1 x (N+2) x (N+2)
        x = x.unfold(2, 3, 1).unfold(3, 3, 1) # bs x 1 x (N+2) x (N+2) -> bs x 1 x N x N x 3 x 3
        y = (x * K).sum(dim=(-2, -1))
        return y


class SmallLinearBlock3ColorsPY(nn.Module):
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
        K = self.KL(image) # 2 x N x N -> 9 x N x N
        return K.mean()


class SmallSMModel3ColorsDnPY(BaseModel):
    def __init__(self, n, num_imgs):
        super().__init__()
        self.n = n
        self.pre = nn.ModuleList()
        self.post = nn.ModuleList()
        for _ in range(n):
            self.pre.append(SmallSMBlock3ColorsPY(num_imgs))
            self.post.append(SmallSMBlock3ColorsPY(num_imgs))
        self.l = SmallSMBlock3ColorsPY(num_imgs)
        self.c0 = nn.ModuleList()
        self.c1 = nn.ModuleList()
        for _ in range(n):
            self.c0.append(SmallLinearBlock3ColorsPY(num_imgs))
            self.c1.append(SmallLinearBlock3ColorsPY(num_imgs))

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

    N = 256
    frame = 100

    file_rhs = os.path.join(DATA_PATH, f"dambreak_N{N}_200", f"div_v_star_{frame}.bin")
    file_flags = os.path.join(DATA_PATH, f"dambreak_N{N}_200", f"flags_{frame}.bin")
    # A = readA_sparse(64, file_A, DIM=2)
    rhs = torch.tensor(load_vector(file_rhs), dtype=torch.float32)
    flags = torch.tensor(convert_to_binary_images(read_flags(file_flags)), dtype=torch.float32)
    # sol = torch.tensor(load_vector(file_sol), dtype=torch.float32)

    # torch.set_grad_enabled(False) # disable autograd globally

    model = SmallSMModel3ColorsDnPY(1).cuda()

    img_shape = (2, N, N)
    rhs_shape = (1, 1, N, N)
    image = flags.reshape(img_shape).cuda()
    x = rhs.reshape(rhs_shape).expand((8,)+ rhs_shape[1:]).cuda()
    x.requires_grad = True
    x1 = rhs.reshape(rhs_shape).expand((8,)+ rhs_shape[1:]).cuda()
    x1.requires_grad = True

    y = model(image, x)
    print(y.shape)
    # torch.set_grad_enabled(False)
    # for _ in range(10):
    #     y = model(image, x)
        # y1 = model1(image, x1)
        # y.sum().backward()
        # y1.sum().backward()

    # torch.cuda.synchronize()

    # iters = 100
    # forward = 0.0
    # backward = 0.0
    # for _ in range(iters):
    #     start = time.time()
    #     y = model(image, x)
    #     torch.cuda.synchronize()
    #     forward += time.time() - start

    #     start = time.time()
    #     y.sum().backward()
    #     torch.cuda.synchronize()
    #     backward += time.time() - start

    # print('PyTorch\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))

    # forward = 0.0
    # backward = 0.0
    # for _ in range(iters):
    #     start = time.time()
    #     y1 = model1(image, x1)
    #     torch.cuda.synchronize()
    #     forward += time.time() - start

    #     start = time.time()
    #     y1.sum().backward()
    #     torch.cuda.synchronize()
    #     backward += time.time() - start

    # print('CUDA kernel\nForward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/iters, backward * 1e6/iters))

    # print((y - y1).abs().max())
    # y.sum().backward()
    # y1.sum().backward()
    # print((model.KL.bias.grad - model1.bias.grad).abs().mean())
    # print((model.KL.weight.grad - model1.weight.grad).abs().mean())
    # print((model.pre[1].KL.weight.grad - model1.pre[1].KL.weight.grad).abs().max())
    # print((x.grad - x1.grad).abs().max())


