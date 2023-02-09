import os, sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from model import FluidNet
from train import train_

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data_fluidnet")
sys.path.insert(1, os.path.join(dir_path, 'lib'))
import read_data as hf

# import conjugate_gradient as cg
# import read_data as hf
class MyDataset(Dataset):
    def __init__(self, N, DIM, flags, A):
        self.N = N
        self.DIM = DIM
        self.n = N**DIM
        self.flags = flags
        self.fluid_cells = torch.where(abs(flags-2) < 1e-15)[0]
        self.A = A
    def __getitem__(self, index):
        x = torch.zeros(self.n)
        x[self.fluid_cells[index]] = 1
        return torch.stack([x, self.flags]).reshape((2,) + (self.N,)*self.DIM), self.A
    def get_rhs(self, index):
        x = torch.zeros(self.n)
        x[self.fluid_cells[index]] = 1
        return x
    def __len__(self):
        return len(self.fluid_cells)


def train(dim, DIM, lr, epoch_num, bs, data_set):

    data_loader = DataLoader(data_set, batch_size=bs, shuffle=True)

    model = FluidNet()
    model.move_to(cuda)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loss, validation_loss, time_history = train_(epoch_num, data_loader, data_loader, model, optimizer)

    outdir = os.path.join(data_path, f"output_single_{DIM}D_{dim}")
    suffix = f"frame_{frame}"
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))

def test(model_file, N, DIM, A, flags, rhs, dcdm_iters=0):
    model = FluidNet()
    # model.to(cuda)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    with torch.no_grad():
        if dcdm_iters == 0:
            y = torch.stack([rhs, flags]).reshape((1, 2,)+(N,)*DIM)
            x_pred = model(y).flatten()
        else:
            x_pred, *_ = fluidnet_dcdm(rhs, A, flags, torch.zeros_like(rhs), model, dcdm_iters, tol=1e-10)
        r = rhs - A @ x_pred
    return x_pred, r

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.sparse.linalg as slin
    dim = 64
    DIM = 2
    dim2 = dim**DIM
    lr = 0.001
    epoch_num = 100
    b_size = 10 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    cuda = torch.device("cuda") # Use CUDA for training

    frame = 800
    A = torch.load(os.path.join(data_path, f"dambreak_{DIM}D_{dim}", "preprocessed", f"A_{frame}.pt"))
    x = torch.load(os.path.join(data_path, f"dambreak_{DIM}D_{dim}", "preprocessed", f"x_{frame}.pt"))
    flags = x[1]
    weight = torch.load(os.path.join(data_path, f"dambreak_{DIM}D_{dim}", "preprocessed", f"w_{frame}.pt"))
    data_set = MyDataset(dim, DIM, flags, A)

    train(dim, DIM, lr, epoch_num, b_size, data_set)


    from cg_tests import fluidnet_dcdm, CG
    rhs = data_set.get_rhs(10)
    # rhs = rhs.to(cuda)
    x_, r_ = test(os.path.join(data_path, f"output_single_{DIM}D_{dim}", f"model_frame_{frame}.pth"),
             dim, DIM, A, flags, rhs, dcdm_iters=15)
    print("FluidNet residual", r_.norm().item())
    plt.imshow(x_.reshape((dim,)*DIM).T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig("pred.png")
    plt.close()
    print(x_.max().item(), x_.min().item())

    A = hf.readA_sparse(os.path.join(data_path, f"dambreak_{DIM}D_{dim}", f"A_{frame}.bin")).astype(np.float32)
    rhs = data_set.get_rhs(10).numpy()
    x, *_ = CG(rhs, A, np.zeros_like(rhs), max_it=1000, verbose=False)
    r = rhs - A @ x
    print("CG residual", np.linalg.norm(r))
    plt.imshow(x.reshape((dim,)*DIM).T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig("gt.png")
    plt.close()
    print(x.max(), x.min())

    plt.imshow(r_.reshape((dim,)*DIM).T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig("residual.png")
