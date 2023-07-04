import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("..")
from sm_model import *


if __name__ == '__main__':
    import time
    torch.set_default_dtype(torch.float64)
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bs = 1
    N = 16
    image = torch.rand(1, N, N, N, device=cuda_device)
    image1 = torch.clone(image)
    x = torch.rand(bs, 1, N, N, N, device=cuda_device, requires_grad=True)

    model = SmallLinearBlock3DPY().to(cuda_device)
    model1 = SmallLinearBlock3D().to(cuda_device)

    y = model(image)
    yy = model1(image1)
    print((y - yy).abs().max())

    # y.sum().backward()
    # yy.sum().backward()

    # print((model.KL.weight.grad - model1.KL.weight.grad).abs().max())
    # print((model.KL.bias.grad - model1.KL.bias.grad).abs().max())
    # print(x.grad)
    model.KL.weight.requires_grad = True
    model.KL.bias.requires_grad = True
    torch.use_deterministic_algorithms(True)
    torch.autograd.gradcheck(SMLinearFunction3D.apply, (image, model.KL.weight, model.KL.bias), nondet_tol=1e-12, fast_mode=True)

    # forward = 0
    # backward = 0
    # for _ in range(10000):
    #     start = time.time()
    #     new_h, new_C = model(image, x)
    #     forward += time.time() - start

    #     start = time.time()
    #     (new_h.sum() + new_C.sum()).backward()
    #     backward += time.time() - start

    # print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))
