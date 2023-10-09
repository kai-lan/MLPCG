import torch

def error_loss(x, x_):
    bs = x.shape[0]
    r = torch.zeros(1, device=x.device)
    for i in range(bs):
        r += (x - x_).norm(1)
    return r / bs
def residual_loss_old(x, y, A):
    bs = x.shape[0]
    r = torch.zeros(1, device=x.device)
    for i in range(bs):
        r_norm = (y[i] - A @ x[i]).norm()
        r += r_norm # No need to compute relative residual because inputs are all unit vectors
    return r / bs
# Please use CSC format for A!
# mean is used for training, and sum is for loss evaluation
def residual_loss(x, y, A, mean=True):
    if A.layout != torch.sparse_csc:
        print("You are not using CSC format, so expect it slow!")
    r = y - x.matmul(A)
    if not mean: return r.norm(dim=1).sum()
    return r.norm(dim=1).mean()
def squared_loss(x, y, A):
    bs = x.shape[0]
    r = torch.zeros(1, device=x.device)
    for i in range(bs):
        r += (y[i] - A @ x[i]).square().sum() # No need to compute relative residual because inputs are all unit vectors
    return r / bs

# Energy loss: negative decreasing
def energy_loss(x, b, A):
    bs = x.shape[0]
    r = torch.zeros(1, device=x.device)
    for i in range(bs):
        r += 0.5 * x[i].dot(A @ x[i]) - x[i].dot(b[i])
    return r / bs
# Scaled loss in 2-norm
def scaled_loss_2(x, y, A): # bs x dim x dim (x dim)
    bs = x.shape[0]
    result = torch.zeros(1, dtype=x.dtype, device=x.device)
    for i in range(bs):
        Ax = A @ x[i]
        alpha = x[i].dot(y[i]) / x[i].dot(Ax)
        r = (y[i] - alpha * Ax).norm() #.square().sum()
        result += r
    return result / bs
def scaled_loss_A(x, y, A): # bs x dim x dim (x dim)
    bs = x.shape[0]
    result = torch.zeros(1, dtype=x.dtype, device=x.device)
    for i in range(bs):
        Ax = A @ x[i]
        alpha = x[i].dot(y[i]) / x[i].dot(Ax)
        r = y[i] - alpha * Ax
        result += r.dot(A @ r)
    return result / bs
