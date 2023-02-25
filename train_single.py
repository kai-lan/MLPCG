import os, sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from model import *
from train import train_
from lib.read_data import load_vector
from lib.dataset import MyDataset
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data_fluidnet")
sys.path.insert(1, os.path.join(dir_path, 'lib'))
import read_data as hf

class ResidualDataset(Dataset):
    def __init__(self, path, shape, flags, A, perm):
        self.path = path
        self.shape = shape
        self.flags = flags
        self.A = A
        self.perm = perm
    def __getitem__(self, index):
        index = self.perm[index]
        x = np.load(self.path + f"/b_res_{index}.npy")
        x = torch.from_numpy(x)
        x = torch.stack([x, self.flags])
        return x.reshape((2,) + self.shape), self.A
    def __len__(self):
        return len(self.perm)

class RitzDataset(Dataset):
    def __init__(self, path, shape, flags, A, perm):
        self.path = path
        self.shape = shape
        self.n = np.prod(shape)
        self.flags = flags
        self.A = A
        self.perm = perm
    def __getitem__(self, index):
        index = self.perm[index]
        x = np.load(self.path + f"/b_dambreak_800_{index}.npy")
        x = torch.from_numpy(x)
        x = torch.stack([x, self.flags])
        return x.reshape((2,) + self.shape), self.A
    def __len__(self):
        return len(self.perm)

class RandomDataset(Dataset):
    def __init__(self, shape, flags, A):
        self.shape = shape
        self.flags = flags
        self.dim2 = np.prod(shape)
        self.A = A
        self.valid_inds = torch.where(flags == 2)[0].numpy()
    def __getitem__(self, index):
        xx = torch.rand(self.dim2, dtype=torch.float32)
        xx[~self.valid_inds] = 0
        xx = torch.stack([xx, self.flags])
        return xx.reshape((2,) + self.shape), self.A
    def __len__(self):
        return len(self.valid_inds)

class StandardDataset(Dataset):
    def __init__(self, shape, flags, A):
        self.shape = shape
        self.flags = flags
        self.dim2 = np.prod(shape)
        self.A = A
        self.valid_inds = torch.where(flags == 2)[0].numpy()
        self.perm = np.random.permutation(self.valid_inds)
    def __getitem__(self, index):
        index = self.perm[index]
        xx = torch.zeros(self.dim2, dtype=torch.float32)
        xx[index] = 1
        xx = torch.stack([xx, self.flags])
        return xx.reshape((2,) + self.shape), self.A
    def __len__(self):
        return len(self.valid_inds)

class RHSDataset(Dataset):
    def __init__(self, x, A):
        self.x = x
        self.A = A
    def __getitem__(self, _):
        return self.x, self.A
    def __len__(self):
        return 1

def train(outdir, suffix, lr, epoch_num, bs, train_set, test_set):
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    model = FluidNet()
    model.move_to(cuda)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loss, validation_loss, time_history = train_(epoch_num, train_loader, test_loader, model, optimizer)


    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))
    return training_loss, validation_loss, time_history

def test(model_file, N, DIM, A, flags, rhs, dcdm_iters=0):
    model = FluidNet()
    model.load_state_dict(torch.load(model_file))
    def fluidnet_predict(r):
        nonlocal model, flags
        with torch.no_grad():
            r = F.normalize(r, dim=0)
            b = torch.stack([r, flags]).view(1, 2, N, N)
            x = model(b).flatten()
        return x
    model.eval()
    with torch.no_grad():
        if dcdm_iters == 0:
            y = torch.stack([rhs, flags]).reshape((1, 2,)+(N,)*DIM)
            x_pred = model(y).flatten()
        else:
            x_pred, *_ = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict, dcdm_iters, tol=1e-10)
        r = rhs - A @ x_pred
    return x_pred, r

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.sparse.linalg as slin
    N = 256
    DIM = 2
    dim2 = N**DIM
    lr = 0.001
    epoch_num = 50
    cuda = torch.device("cuda") # Use CUDA for training

    frame = 800
    A = torch.load(os.path.join(data_path, f"dambreak_{DIM}D_{N}", "preprocessed", f"A_{frame}.pt"))
    x = torch.load(os.path.join(data_path, f"dambreak_{DIM}D_{N}", "preprocessed", f"x_{frame}.pt"))
    gt = torch.tensor(load_vector(os.path.join(data_path, f"dambreak_{DIM}D_{N}", f"pressure_{frame}.bin")))
    flags = x[1]

    num_rhs = 150
    perm = np.random.permutation(range(num_rhs))
    training_set = ResidualDataset(os.path.join(dir_path, "data_dcdm", f"train_{DIM}D_{N}"), (N,)*DIM, flags, A, perm)
    # training_set = RitzDataset(os.path.join(dir_path, "data_dcdm", f"train_{DIM}D_{N}"), (N,)*DIM, flags, A, perm)
    # training_set = RandomDataset((N,)*DIM, flags, A)
    # training_set = StandardDataset((N,)*DIM, flags, A)
    # training_set = RHSDataset(x.reshape((2,)+(N,)*DIM), A)

    # validation_set = RitzDataset(os.path.join(dir_path, "data_dcdm", f"train_{DIM}D_{N}"), (N,)*DIM, flags, A, perm[256:])
    validation_set = RHSDataset(x.reshape((2,)+(N,)*DIM), A)

    outdir = os.path.join(dir_path, "data_fluidnet", f"output_single_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    suffix = f"frame_{frame}_res"
    training_loss, validation_loss, time_history = train(outdir, suffix, lr, epoch_num, 25, training_set, validation_set)


    # xx = np.load(os.path.join(dir_path, "data_dcdm", f"train_{DIM}D_{N}", f"b_dambreak_800_11.npy"))
    # xx = torch.from_numpy(xx)
    # xx = torch.stack([xx, flags])



    from cg_tests import dcdm, CG

    rhs = x[0]
    x_, r_ = test(os.path.join(outdir, f"model_{suffix}.pth"),
             N, DIM, A, flags, rhs, dcdm_iters=20)
    print("Simple model residual", r_.norm().item()/rhs.norm().item())

    A = hf.readA_sparse(os.path.join(data_path, f"dambreak_{DIM}D_{N}", f"A_{frame}.bin")).astype(np.float32)
    # rhs = data_set.get_rhs(10).numpy()
    rhs = hf.load_vector(os.path.join(data_path, f"dambreak_{DIM}D_{N}", f"div_v_star_{frame}.bin"))
    x, res_history = CG(rhs, A, np.zeros_like(rhs), max_it=1000, tol=1e-5, verbose=False)
    r = rhs - A @ x
    print(f"CG residual after {len(res_history)} iterations", np.linalg.norm(r)/np.linalg.norm(rhs))


    fig, axes = plt.subplots(2)
    axes[0].plot(training_loss, label='training')
    axes[1].plot(validation_loss, label='validation')
    plt.savefig("loss.png", bbox_inches='tight')
