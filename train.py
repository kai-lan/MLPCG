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
def train_(epoch_num, train_loader, validation_loader, model, optimizer):
    training_loss = []
    validation_loss = []
    time_history = []
    t0 = time.time()
    for i in range(1, epoch_num+1):
        print(f"Training at {i} / {epoch_num}")
        t0 = time.time()
        # Training
        los = 0.0
        for ii, (data, A) in enumerate(tqdm(train_loader), 1):# x: (bs, dim, dim, dim)
            data = data.to(model.device)
            A = A.to(model.device)
            x_pred = model(data) # (bs, 2, N, N)
            loss = model.loss(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            los += loss.item()
        # Validation
        tot_loss_train, tot_loss_val = 0, 0
        with torch.no_grad():
            for data, A in train_loader:
                data = data.to(model.device)
                A = A.to(model.device)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_train += model.loss(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
            for data, A in validation_loader:
                data = data.to(model.device)
                A = A.to(model.device)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_val += model.loss(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(training_loss[-1], validation_loss[-1])
    return training_loss, validation_loss, time_history


if __name__ == '__main__':
    # command variables
    N = 256
    DIM = 2
    lr = 1.0e-4
    bc = 'dambreak'
    include_bd = False
    epoch_num = 200
    b_size = 25 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_data_points = 950 # number of images: 1000 total, leaving 50 for testing later
    cuda = torch.device("cuda") # Use CUDA for training
    prefix = '_' if include_bd else ''
    log = LoggingWriter()
    log.record({"N": N,
                "DIM": DIM,
                "bc": bc,
                "include bd": include_bd,
                "lr": lr,
                "Epoches": epoch_num,
                "batch size": b_size,
                "Tot data points": total_data_points})

    resume = False

    model = FluidNet()
    model.move_to(cuda)
    if resume:
        model.load_state_dict(torch.load(os.path.join(fluidnet_data_path, f"{prefix}output_{DIM}D_{N}", f"model_{bc}.pth")))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    perm = np.random.permutation(total_data_points) + 1
    train_size = round(0.8 * total_data_points)
    train_set = MyDataset(os.path.join(fluidnet_data_path, f"{prefix}{bc}_{DIM}D_{N}/preprocessed"), ['x_*.pt', 'A_*.pt'], perm[:train_size], (2,)+(N,)*DIM)
    validation_set = MyDataset(os.path.join(fluidnet_data_path, f"{prefix}{bc}_{DIM}D_{N}/preprocessed"), ['x_*.pt', 'A_*.pt'],perm[train_size:], (2,)+(N,)*DIM)


    outdir = os.path.join(fluidnet_data_path, f"{prefix}output_{DIM}D_{N}")
    suffix = bc + str(epoch_num)

    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=b_size, shuffle=False)

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
    plt.savefig(f"loss_{bc}.png", bbox_inches='tight')
