import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))
dcdm_data_path = os.path.join(dir_path, "data_dcdm")
fluidnet_data_path = os.path.join(dir_path, "data_fluidnet")
# sys.path.insert(1, os.path.join(dir_path, 'lib'))
import lib.read_data as hf
from lib.write_log import LoggingWriter
from lib.dataset import *
from model import DCDM, FluidNet

# Fluidnet
from train import train_

# DCDM
def train(epoch_num, train_loader, validation_loader, model, optimizer):
    training_loss = []
    validation_loss = []
    time_history = []
    t0 = time.time()
    for i in range(1, epoch_num+1):
        print(f"Training at {i} / {epoch_num}")
        t0 = time.time()
        tot_loss_train, tot_loss_val = 0, 0
        # Training
        for ii, (x, *_) in enumerate(tqdm(train_loader), 1):# x: (bs, dim, dim, dim)
            x = x.to(model.device)
            y_pred = model(x) # input: (bs, 1, dim, dim, dim)
            loss = model.loss(y_pred, x, A_sparse)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        with torch.no_grad():
            for x, *_ in train_loader:
                x = x.to(model.device)
                y_pred = model(x) # input: (bs, 1, dim, dim, dim)
                tot_loss_train += model.loss(y_pred, x, A_sparse).item()
            for x, *_ in validation_loader:
                x = x.to(model.device)
                y_pred = model(x) # input: (bs, 1, dim, dim, dim)
                tot_loss_val += model.loss(y_pred, x, A_sparse).item()
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(training_loss[-1], validation_loss[-1])
    return training_loss, validation_loss, time_history

class MyDataset_(Dataset):
    def __init__(self, data_folder, perm, shape, A, include_bd) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.shape = shape
        self.flags = torch.ones((N,)*DIM) * 2
        if include_bd:
            self.flags[0] = 0
            self.flags[-1] = 0
            self.flags[:, 0] = 0
            self.flags[:, -1] = 0
        self.flags = self.flags.flatten()
        self.A = A
        self.perm = perm
    def __getitem__(self, index):
        index = perm[index]
        file = os.path.join(self.data_folder, f"b_empty_{index}.npy")
        x = torch.from_numpy(np.load(file)).float()
        return torch.stack([x, self.flags]).view(self.shape), self.A
    def __len__(self):
        return len(self.perm)

if __name__ == '__main__':
    # command variables
    N = 64
    DIM = 2
    include_bd = False
    bc = 'empty'
    prefix = '_' if include_bd else ''
    lr = 1.0e-4
    epoch_num = 200
    b_size = 25 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_data_points = 1000 # 1000 is better than 2000
    cuda = torch.device("cuda") # Use CUDA for training

    log = LoggingWriter()
    log.record({"N": N,
                "DIM": DIM,
                "bc": bc,
                "lr": lr,
                "Epoches": epoch_num,
                "batch size": b_size,
                "Tot data points": total_data_points})

    dcdm = False
    fluidnet = True
    resume = False

    if dcdm:
        name_sparse_matrix = os.path.join(dcdm_data_path, f"{prefix}train_{DIM}D_{N}/A_empty.bin")
        A_sparse_scipy = hf.readA_sparse(name_sparse_matrix, 'f')
        A_sparse = torch.sparse_csr_tensor(A_sparse_scipy.indptr, A_sparse_scipy.indices, A_sparse_scipy.data, A_sparse_scipy.shape, dtype=torch.float32, device=cuda)

        model = DCDM(DIM)
        model.move_to(cuda)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        perm = np.random.permutation(total_data_points)
        train_size = round(0.8 * total_data_points)
        train_set = MyDataset(os.path.join(dcdm_data_path, f"{prefix}train_{DIM}D_{N}"), [f"b_{bc}_*.npy"], perm[:train_size], (1,)+(N,)*DIM)
        validation_set = MyDataset(os.path.join(dcdm_data_path, f"{prefix}train_{DIM}D_{N}"), [f"b_{bc}_*.npy"], perm[train_size:], (1,)+(N,)*DIM)

        outdir = os.path.join(dcdm_data_path, f"{prefix}output_{DIM}D_{N}")
        # suffix = time.ctime().replace(' ', '-')
        suffix = 'empty'

    elif fluidnet:
        model = FluidNet()
        model.move_to(cuda)
        if resume:
            model.load_state_dict(torch.load(os.path.join(fluidnet_data_path, f"{prefix}output_{DIM}D_{N}", "model_.pth")))
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # perm = np.random.permutation(total_data_points) + 1
        perm = np.random.permutation(total_data_points)
        train_size = round(0.8 * total_data_points)
        # train_set = MyDataset(os.path.join(fluidnet_data_path, f"dambreak_{DIM}D_{N}/preprocessed"), ['x_*.pt', 'A_*.pt'], perm[:train_size], (2,)+(N,)*DIM)
        # validation_set = MyDataset(os.path.join(fluidnet_data_path, f"dambreak_{DIM}D_{N}/preprocessed"), ['x_*.pt', 'A_*.pt'],perm[train_size:], (2,)+(N,)*DIM)

        name_sparse_matrix = os.path.join(dcdm_data_path, f"{prefix}train_{DIM}D_{N}/A_{bc}.bin")
        A_sparse_scipy = hf.readA_sparse(name_sparse_matrix, 'f')
        A = torch.sparse_csr_tensor(A_sparse_scipy.indptr, A_sparse_scipy.indices, A_sparse_scipy.data, A_sparse_scipy.shape, dtype=torch.float32, device=cuda).to_sparse_coo()
        train_set = MyDataset_(os.path.join(dcdm_data_path, f"{prefix}train_{DIM}D_{N}"), perm[:train_size], (2,)+(N,)*DIM, A, include_bd)
        validation_set = MyDataset_(os.path.join(dcdm_data_path, f"{prefix}train_{DIM}D_{N}"), perm[train_size:], (2,)+(N,)*DIM, A, include_bd)

        outdir = os.path.join(fluidnet_data_path, f"{prefix}output_{DIM}D_{N}")
        suffix = 'empty'

    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=b_size, shuffle=False)

    if dcdm:
        training_loss, validation_loss, time_history = train(epoch_num, train_loader, validation_loader, model, optimizer)
    elif fluidnet:
        training_loss, validation_loss, time_history = train_(epoch_num, train_loader, validation_loader, model, optimizer)

    os.makedirs(outdir, exist_ok=True)
    log.write(os.path.join(outdir, f"settings_{suffix}.log"))
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    # Save model
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2)
    axes[0].plot(training_loss, label='training')
    axes[1].plot(validation_loss, label='validation')
    plt.savefig("loss.png", bbox_inches='tight')
