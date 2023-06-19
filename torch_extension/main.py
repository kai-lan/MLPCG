import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.utils.cpp_extension import load
# lltm = load(name='lltm', sources=['sm_block.cpp', 'sm_block_kernel.cu'])


# class LLTMFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weights, bias, old_h, old_cell):
#         outputs = lltm.forward(input, weights, bias, old_h, old_cell)
#         variables = outputs[1:] + [weights]
#         ctx.save_for_backward(*variables)

#         return new_h, new_cell

#     @staticmethod
#     def backward(ctx, grad_h, grad_cell):
#         outputs = lltm.backward(
#             grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
#         d_old_h, d_input, d_weights, d_bias, d_old_cell, *_ = outputs
#         return d_input, d_weights, d_bias, d_old_h, d_old_cell


# class LLTM(torch.nn.Module):
#     def __init__(self, input_features, state_size):
#         super(LLTM, self).__init__()
#         self.input_features = input_features
#         self.state_size = state_size
#         self.weights = torch.nn.Parameter(
#             torch.empty(3 * state_size, input_features + state_size))
#         self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.state_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, +stdv)

#     def forward(self, input, state):
#         return LLTMFunction.apply(input, self.weights, self.bias, *state)

class SMBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, x, weights, bias):
        N = x.shape[-1]
        y = torch.zeros_like(x)
        image = F.pad(image, (1,)*4)
        x = F.pad(x, (1,)*4)
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
        # print(grad_output.shape)
        image, x, = ctx.saved_tensors
        grad_w = torch.zeros(9, 1, 3, 3)
        grad_b = torch.zeros(9)
        N = x.shape[-1] - 2
        for p in range(9):
            k, l = p//3, p%3
            for m in range(3):
                for n in range(3):
                    var = image[:, m:m+N, n:n+N] * x[:, :, k:k+N, l:l+N]
                    grad_w[p, 0, m, n] = (var * grad_output).sum()
            grad_b[p] = (x[:, :, k:k+N, l:l+N] * grad_output).sum()

        return None, None, grad_w, grad_b

class SmallSMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(9, 1, 3, 3))
        self.bias = nn.Parameter(torch.ones(9))
        # self.KL = nn.Conv2d(1, 9, kernel_size=3, padding='same', bias=True)
        # print(self.KL.weight.shape, self.KL.bias.shape, self.weights.shape)
    def forward(self, image, x):
        return SMBlockFunction.apply(image, x, self.weights, self.bias)
    # def forward(self, image, x): # 1 x N x N, bs x 1 x N x N
    #     K = self.KL(image) # 1 x N x N -> 9 x N x N
    #     K = K.permute((1, 2, 0)) # 9 x N x N -> N x N x 9
    #     K = K.unflatten(2, (3, 3)) # N x N x 9 -> N x N x 3 x 3
    #     x = F.pad(x, (1, 1, 1, 1)) # bs x 1 x N x N -> bs x 1 x (N+2) x (N+2)
    #     x = x.unfold(2, 3, 1).unfold(3, 3, 1) # bs x 1 x (N+2) x (N+2) -> bs x 1 x N x N x 3 x 3
    #     y = (x * K).sum(dim=(-2, -1))
    #     return y

if __name__ == '__main__':
    import time
    torch.set_default_dtype(torch.float64)
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")

    bs = 5
    N = 6
    image = torch.rand(1, N, N)
    x = torch.rand(bs, 1, N, N)

    model = SmallSMBlock()

    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    y = model(image, x)
    # L = y.sum()
    # L.backward()
    # print(model.weights.grad)
    torch.autograd.gradcheck(SMBlockFunction.apply, (image, x, model.weights, model.bias))
    # optimizer.step()
    # SmBlockFunction.apply()
    # rnn = LLTM(input_features, state_size).to(cuda_device)

    # forward = 0
    # backward = 0
    # for _ in range(10000):
    #     start = time.time()
    #     new_h, new_C = rnn(X, (h, C))
    #     forward += time.time() - start

    #     start = time.time()
    #     (new_h.sum() + new_C.sum()).backward()
    #     backward += time.time() - start

    # print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))