import os, sys
sys.path.insert(1, 'lib')
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from lib.write_log import LoggingWriter
from lib.dataset import *
from lib.GLOBAL_VARS import *
from model import *


# Fluidnet
def train_(A, epoch_num, train_loader, validation_loader, model, optimizer, loss_fn):
    A = A.to(model.device)
    training_loss = []
    validation_loss = []
    time_history = []
    t0 = time.time()
    for i in range(1, epoch_num+1):
        t0 = time.time()
        # Training
        los = 0.0
        for ii, data in enumerate(train_loader, 1):# x: (bs, dim, dim, dim)
            data = data.to(model.device)
            x_pred = model(data) # (bs, 2, N, N)
            loss = loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            los += loss.item()
        # Validation
        tot_loss_train, tot_loss_val = 0, 0
        with torch.no_grad():
            for data in train_loader:
                data = data.to(model.device)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_train += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
            for data in validation_loader:
                data = data.to(model.device)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_val += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(training_loss[-1], validation_loss[-1], f"({i} / {epoch_num})")
    return training_loss, validation_loss, time_history

def saveData(model, optimizer, epoch, log, outdir, suffix, train_loss, valid_loss, time_history):
    if log is not None:
        log.record({"N": N,
                    "DIM": DIM,
                    "bc": bc,
                    "lr": lr,
                    "Epoches per matrix": epoch_num_per_matrix,
                    "Epoches": epoch,
                    "batch size": b_size,
                    "Num matrices": total_matrices,
                    "Num RHS": num_rhs})
        log.write(os.path.join(outdir, f"settings_{suffix}.log"))
    # np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), train_loss)
    # np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), valid_loss)
    # np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': train_loss,
            'validation_loss': valid_loss,
            'time': time_history
            }, os.path.join(outdir, f"checkpt_{suffix}.tar"))

def loadData(outdir, suffix):
    checkpt = torch.load(os.path.join(outdir, f"checkpt_{suffix}.tar"))
    epoch = checkpt['epoch']
    training_loss = checkpt['training_loss']
    validation_loss = checkpt['validation_loss']
    time_history = checkpt['time']
    model_params = checkpt['model_state_dict']
    optim_params = checkpt['optimizer_state_dict']
    return epoch, list(training_loss), list(validation_loss), list(time_history), model_params, optim_params

if __name__ == '__main__':
    np.random.seed(2) # same random seed for debug
    N = 256
    DIM = 2
    lr = 1.0e-4
    epoch_num_per_matrix = 5
    epoch_num = 50
    bc = 'dambreak'
    b_size = 20 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_matrices = 50 # number of matrices chosen for training
    num_ritz = 1000
    num_rhs = 200 # number of ritz vectors for training for each matrix
    kernel_size = 3 # kernel size
    cuda = torch.device("cuda") # Use CUDA for training

    log = LoggingWriter()

    resume = False
    loss_type = 'scaledA'

    if loss_type == 'res':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_res'
        loss_fn = "model.residual_loss"
    elif loss_type == 'eng':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_eng'
        loss_fn = "model.energy_loss"
    elif loss_type == 'scaled2':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_scaled2'
        loss_fn = "model.scaled_loss_2"
    elif loss_type == 'scaledA':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_scaledA'
        loss_fn = "model.scaled_loss_A"
    else:
        raise Exception("No such loss type")

    outdir = os.path.join(OUT_PATH, f"output_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    inpdir = os.path.join(DATA_PATH, f"{bc}_N{N}_200/preprocessed")

    model = FluidNet(kernel_size)
    model.move_to(cuda)
    loss_fn = eval(loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume:
        ep, train_loss, valid_loss, time_history, model_params, optim_params = loadData(outdir, suffix)
        model.load_state_dict(model_params)
        optimizer.load_state_dict(optim_params)
        start_epoch = ep
    else:
        train_loss, valid_loss, time_history = [], [], []
        start_epoch = 0


    train_size = round(0.8 * num_rhs)
    perm = np.random.permutation(num_rhs)
    train_set = MyDataset(None, perm[:train_size], (2,)+(N,)*DIM)
    valid_set = MyDataset(None, perm[train_size:], (2,)+(N,)*DIM)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False)
    matrices = np.random.permutation(range(1, 201))[:total_matrices]
    np.save(f"{outdir}/matrices_trained_{total_matrices}.npy", matrices)
    # matrices = range(1, total_matrices+1)
    start_time = time.time()
    for i in range(1, epoch_num+1):
        print(f"Epoch: {i}/{epoch_num}")
        tl, vl = 0.0, 0.0
        for j in matrices:
            print('Matrix', j)
            train_set.data_folder = os.path.join(f"{inpdir}/{j}")
            valid_set.data_folder = os.path.join(f"{inpdir}/{j}")
            A = torch.load(f"{train_set.data_folder}/A.pt")
            _training_loss, _validation_loss, _= train_(A, epoch_num_per_matrix, train_loader, valid_loader, model, optimizer, loss_fn)
            tl += np.sum(_training_loss)
            vl += np.sum(_validation_loss)
        train_loss.append(tl)
        valid_loss.append(vl)
        time_history.append(time.time() - start_time)
        saveData(model, optimizer, i+start_epoch, log, outdir, suffix, train_loss, valid_loss, time_history)
    end_time = time.time()
    print("Took", end_time-start_time, 's')

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2)
    axes[0].plot(train_loss, label='training')
    axes[1].plot(valid_loss, label='validation')
    plt.savefig(f"loss_{bc}.png", bbox_inches='tight')
