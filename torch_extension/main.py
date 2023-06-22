import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import smblock

# from torch.utils.cpp_extension import load
# lltm = load(name='lltm', sources=['sm_block.cpp', 'sm_block_kernel.cu'])


class SMBlockFunctionPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, x, weights, bias):
        N = x.shape[-1]
        y = torch.zeros_like(x)
        image = F.pad(image, (1,)*4) # 1, N+2, N+2
        x = F.pad(x, (1,)*4) # bs, 1, N+2, N+2

        for i in range(1, N+1):
            for j in range(1, N+1):
                K = ((weights.squeeze() * image[0, i-1:i+2, j-1:j+2])).sum(dim=(1, 2)) + bias
                K = K.view((3, 3))
                xx = x[:, 0, i-1:i+2, j-1:j+2]
                y[:, 0, i-1, j-1] = (xx * K).sum(dim=(-2, -1))

        ctx.save_for_backward(image, x,)
        return y
    @staticmethod
    def backward(ctx, grad_output): # return the same number of outputs as forward function arguments
        image, x, = ctx.saved_tensors
        grad_w = torch.zeros(9, 1, 3, 3, device=image.device)
        grad_b = torch.zeros(9, device=image.device)
        N = x.shape[-1] - 2
        for p in range(9):
            k, l = p//3, p%3
            for m in range(3):
                for n in range(3):
                    var = image[:, m:m+N, n:n+N] * x[:, :, k:k+N, l:l+N]
                    grad_w[p, 0, m, n] = (var * grad_output).sum()
            grad_b[p] = (x[:, :, k:k+N, l:l+N] * grad_output).sum()
        return None, None, grad_w, grad_b

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
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(9, 1, 3, 3))
        self.bias = nn.Parameter(torch.ones(9))
        # self.KL = nn.Conv2d(1, 9, kernel_size=3, padding='same', bias=True)
        # print(self.KL.weight.shape, self.KL.bias.shape, self.weights.shape)
    def forward(self, image, x):
        return SMBlockFunction.apply(image, x, self.weights, self.bias)
    def test(self, image, x):
        return SMBlockFunctionPY.apply(image, x, self.weights, self.bias)


if __name__ == '__main__':
    import time
    torch.set_default_dtype(torch.float64)
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bs = 2
    N = 5
    image = torch.rand(1, N, N, device=cuda_device)
    x = torch.rand(bs, 1, N, N, device=cuda_device, requires_grad=True)
    model = SmallSMBlock().to(cuda_device)

    y = model(image, x)
    yy = model.test(image, x)
    print((y - yy).abs().max())

    # print(y)
    # L = y.sum()
    # L.backward()
    # print(model.weights.grad)
    # print(x.grad)

    torch.autograd.gradcheck(SMBlockFunction.apply, (image, x, model.weights, model.bias))

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
