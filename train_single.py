import os, sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from model import *
from train import train_, saveData
from lib.read_data import *
from lib.dataset import MyDataset
from lib.write_log import LoggingWriter
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data_fluidnet")
sys.path.insert(1, os.path.join(dir_path, 'lib'))


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
        # x = np.load(self.path + f"/b_{index}_new.npy")
        # x = torch.from_numpy(x)
        x = torch.load(self.path + f"/x_{index}.pt")
        # x = torch.stack([x, self.flags])
        return x.reshape((2,) + self.shape), self.A
    def __len__(self):
        return len(self.perm)

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
    loss_fn = model.inv_energy_loss
    # loss_fn = model.residual_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loss, validation_loss, time_history = train_(epoch_num, train_loader, test_loader, model, optimizer, loss_fn)

    os.makedirs(outdir, exist_ok=True)
    log = LoggingWriter()
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))
    return training_loss, validation_loss, time_history

def test(model, N, DIM, A, flags, rhs, dcdm_iters=0):
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
    # train in single or double precision
    torch.set_default_dtype(torch.float32)
    N = 64
    DIM = 2
    dim2 = N**DIM
    lr = 0.001
    epoch_num = 200
    cuda = torch.device("cuda") # Use CUDA for training

    frame = 800
    A = torch.load(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"preprocessed/{frame}/A.pt"))
    rhs = torch.load(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"preprocessed/{frame}/rhs.pt"))
    flags = torch.load(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"preprocessed/{frame}/flags.pt"))
    gt = torch.tensor(load_vector(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"pressure_{frame}.bin")))

    num_rhs = 500
    perm = np.random.permutation(range(num_rhs))

    training_set = RitzDataset(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}/preprocessed/{frame}"), (N,)*DIM, flags, A, perm)

    validation_set = RHSDataset(torch.stack([rhs, flags]).reshape((2,)+(N,)*DIM), A)

    outdir = os.path.join(OUT_PATH, f"output_single_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    suffix = f"frame_{frame}_downsample"
    #### Train
    # training_loss, validation_loss, time_history = train(outdir, suffix, lr, epoch_num, 10, training_set, validation_set)


    from cg_tests import dcdm, CG
    # outdir = os.path.join(OUT_PATH, f"output_{DIM}D_{N}")
    suffix = f"frame_{frame}_downsample"
    #### Test
    model = NewModel()
    model.load_state_dict(torch.load(os.path.join(outdir, f"model_{suffix}.pth")))
    x_, r_ = test(model, N, DIM, A, flags, rhs, dcdm_iters=20)
    print("Fluidnet", r_.norm().item()/rhs.norm().item())

    A = readA_sparse(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"A_{frame}.bin")).astype(np.float32)
    rhs = load_vector(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"div_v_star_{frame}.bin")).astype(np.float32)
    x, res_history = CG(rhs, A, np.zeros_like(rhs), max_it=1000, tol=1e-5, verbose=False)
    r = rhs - A @ x
    print(f"CG residual after {len(res_history)} iterations", np.linalg.norm(r)/np.linalg.norm(rhs))


    # fig, axes = plt.subplots(2)
    # axes[0].plot(training_loss, label='training')
    # axes[1].plot(validation_loss, label='validation')
    # plt.savefig("loss.png", bbox_inches='tight')
