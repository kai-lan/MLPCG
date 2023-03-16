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
from model import DCDM, FluidNet


# Fluidnet
def train_(epoch_num, train_loader, validation_loader, model, optimizer, loss_fn):
    training_loss = []
    validation_loss = []
    time_history = []
    t0 = time.time()
    for i in range(1, epoch_num+1):
        t0 = time.time()
        # Training
        los = 0.0
        for ii, (data, A) in enumerate(train_loader, 1):# x: (bs, dim, dim, dim)
            data = data.to(model.device)
            A = A.to(model.device)
            x_pred = model(data) # (bs, 2, N, N)
            loss = loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A)
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
                tot_loss_train += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
            for data, A in validation_loader:
                data = data.to(model.device)
                A = A.to(model.device)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_val += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(training_loss[-1], validation_loss[-1], f"({i} / {epoch_num})")
    return training_loss, validation_loss, time_history

def saveData(epoch, log, outdir, suffix, train_loss, valid_loss, time_history):
    log.record({"N": N,
                "DIM": DIM,
                "bc": bc,
                "lr": lr,
                "Epoches per matrix": epoch_num_per_matrix,
                "Epoches": epoch,
                "batch size": b_size,
                "Num matrices": total_matrices,
                "Num Ritz Vectors": num_ritz_vecs})
    log.write(os.path.join(outdir, f"settings_{suffix}.log"))
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), train_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), valid_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))

if __name__ == '__main__':
    np.random.seed(2) # same random seed for debug
    N = 64
    DIM = 2
    lr = 1.0e-3
    epoch_num_per_matrix = 5
    epoch_num = 20
    bc = 'dambreak'
    b_size = 20 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_matrices = 100 # number of matrices chosen for training
    num_ritz_vecs = 500 # number of ritz vectors for training for each matrix
    cuda = torch.device("cuda") # Use CUDA for training
    log = LoggingWriter()


    resume = False
    suffix = bc
    # suffix = bc+'_direction'
    suffix = bc+'_rhs_init'
    outdir = os.path.join(OUT_PATH, f"output_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)

    model = FluidNet()
    model.move_to(cuda)
    loss_fn = model.loss
    # loss_fn = model.direction_loss

    if resume:
        model.load_state_dict(torch.load(os.path.join(OUT_PATH, f"output_{DIM}D_{N}", f"model_{suffix}.pth")))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_size = round(0.8 * num_ritz_vecs)
    perm = np.random.permutation(num_ritz_vecs)
    train_set = MyDataset(None, ['x_*_new.pt', 'A.pt'], perm[:train_size], (2,)+(N,)*DIM)
    valid_set = MyDataset(None, ['x_*_new.pt', 'A.pt'], perm[train_size:], (2,)+(N,)*DIM)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False)
    matrices = np.random.permutation(range(1, 1001))[:total_matrices]
    np.save(f"{outdir}/matrices_trained.npy", matrices)
    # matrices = range(1, total_matrices+1)
    train_loss, valid_loss, time_history = [], [], []
    start_time = time.time()
    for i in range(1, epoch_num+1):
        print(f"Epoch: {i}/{epoch_num}")
        tl, vl = 0.0, 0.0
        for j in matrices:
            print('Matrix', j)
            train_set.data_folder = os.path.join(DATA_PATH, f"{bc}_{DIM}D_{N}/preprocessed/{j}")
            valid_set.data_folder = os.path.join(DATA_PATH, f"{bc}_{DIM}D_{N}/preprocessed/{j}")
            _training_loss, _validation_loss, _= train_(epoch_num_per_matrix, train_loader, valid_loader, model, optimizer, loss_fn)
            tl += np.sum(_training_loss)
            vl += np.sum(_validation_loss)
        train_loss.append(tl)
        valid_loss.append(vl)
        time_history.append(time.time() - start_time)
        saveData(i, log, outdir, suffix, train_loss, valid_loss, time_history)
    end_time = time.time()
    print("Took", end_time-start_time, 's')

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2)
    axes[0].plot(train_loss, label='training')
    axes[1].plot(valid_loss, label='validation')
    plt.savefig(f"loss_{bc}.png", bbox_inches='tight')
