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
from sm_model import *
import matplotlib.pyplot as plt

def move_data(data, device):
    if len(data) > 0:
        for i in range(len(data)):
            data[i] = data[i].to(device)
    else:
        data[:] = data.to(device)

def validation(train_loader, validation_loader, model, loss_fn, image, A):
    tot_loss_train, tot_loss_val = 0, 0
    with torch.no_grad():
        for data in train_loader:
            data = data.to(model.device)
            x_pred = model(image, data) # input: (bs, 1, dim, dim)
            tot_loss_train += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
        for data in validation_loader:
            data = data.to(model.device)
            x_pred = model(image, data) # input: (bs, 1, dim, dim)
            tot_loss_val += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
    return tot_loss_train, tot_loss_val

def train_(image, A, epoch_num, train_loader, validation_loader, model, optimizer, loss_fn):
    A = A.to(model.device)
    image = image.to(model.device)
    training_loss = []
    validation_loss = []
    time_history = []
    grad_history = []
    update_history = []
    t0 = time.time()

    tot_loss_train, tot_loss_val = validation(train_loader, validation_loader, model, loss_fn, image, A)
    training_loss.append(tot_loss_train)
    validation_loss.append(tot_loss_val)
    time_history.append(time.time() - t0)
    old_param = None
    with torch.no_grad():
        for k, param in enumerate(model.parameters()):
            if k == 0:  old_param = param.data.ravel()
            else:       old_param = torch.cat([old_param, param.data.ravel()])
    print(training_loss[-1], validation_loss[-1], f"(0 / {epoch_num})")


    for i in range(1, epoch_num+1):
        # Training
        for ii, data in enumerate(train_loader, 1):# data: (bs, 2, N, N)
            data = data.to(model.device)
            x_pred = model(image, data)
            x_pred = x_pred.squeeze(dim=1).flatten(1) # (bs, 1, N, N) -> (bs, N, N) -> (bs, N*N)
            data = data.squeeze(dim=1).flatten(1)
            loss = loss_fn(x_pred, data, A)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dLdw = 0.0
        new_param = None
        with torch.no_grad():
            for k, param in enumerate(model.parameters()):
                dLdw += param.grad.norm().item()
                if k == 0:  new_param = param.data.ravel()
                else:       new_param = torch.cat([new_param, param.data.ravel()])
        grad_history.append(dLdw)
        update_history.append((new_param - old_param).norm().item())
        old_param = new_param

        # Validation
        tot_loss_train, tot_loss_val = validation(train_loader, validation_loader, model, loss_fn, image, A)
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(training_loss[-1], validation_loss[-1], f"grad {dLdw}" f"({i} / {epoch_num})")

    return training_loss, validation_loss, time_history, grad_history, update_history

def saveData(model, optimizer, epoch, log, outdir, suffix, train_loss, valid_loss, time_history, grad_history, update_history):
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
            'time': time_history,
            'grad': grad_history,
            'update': update_history
            }, os.path.join(outdir, f"checkpt_{suffix}.tar"))

def loadData(outdir, suffix):
    checkpt = torch.load(os.path.join(outdir, f"checkpt_{suffix}.tar"))
    epoch = checkpt['epoch']
    model_params = checkpt['model_state_dict']
    optim_params = checkpt['optimizer_state_dict']
    training_loss = checkpt['training_loss']
    validation_loss = checkpt['validation_loss']
    time_history = checkpt['time']
    grad_history = checkpt['grad']
    update_history = checkpt['update']
    return epoch, model_params, optim_params, list(training_loss), list(validation_loss), list(time_history), list(grad_history), list(update_history)

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    N = 256
    DIM = 2
    lr = 1.0e-3
    epoch_num_per_matrix = 5
    epoch_num = 50
    bc = 'dambreak'
    b_size = 20 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_matrices = 50 # number of matrices chosen for training
    num_ritz = 800
    num_rhs = 400 # number of ritz vectors for training for each matrix
    kernel_size = 3 # kernel size
    cuda = torch.device("cuda") # Use CUDA for training

    log = LoggingWriter()

    resume = False
    loss_type = 'res'
    image_type = 'flags' # flags, ppc, levelset

    if loss_type == 'res':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_res_{image_type}'
        loss_fn = "model.residual_loss"
    elif loss_type == 'eng':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_eng_{image_type}'
        loss_fn = "model.energy_loss"
    elif loss_type == 'scaled2':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_scaled2_{image_type}'
        loss_fn = "model.scaled_loss_2"
    elif loss_type == 'scaledA':
        suffix = f'{bc}_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_scaledA_{image_type}'
        loss_fn = "model.scaled_loss_A"
    else:
        raise Exception("No such loss type")

    suffix += '_smmodeld3'
    outdir = os.path.join(OUT_PATH, f"output_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    inpdir = os.path.join(DATA_PATH, f"{bc}_N{N}_200/preprocessed")

    model = SmallSMModelD2()
    model.move_to(cuda)
    loss_fn = eval(loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume:
        ep, model_params, optim_params, train_loss, valid_loss, time_history, grad_history, update_history = loadData(outdir, suffix)
        model.load_state_dict(model_params)
        optimizer.load_state_dict(optim_params)
        start_epoch = ep
    else:
        train_loss, valid_loss, time_history, grad_history, update_history = [], [], [], [], []
        start_epoch = 0


    train_size = round(0.8 * num_rhs)
    perm = np.random.permutation(num_rhs)

    def transform(x):
        x = x.reshape((1, N, N))
        return x

    train_set = MyDataset(None, perm[:train_size], transform, denoised=False)
    valid_set = MyDataset(None, perm[train_size:], transform, denoised=False)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False)
    matrices = np.random.permutation(range(1, 201))[:total_matrices]
    # matrices = np.load(os.path.join(DATA_PATH, f"{bc}_N{N}_200/train_mat.npy"))
    np.save(f"{outdir}/matrices_trained_{total_matrices}.npy", matrices)
    start_time = time.time()

    for i in range(1, epoch_num+1):
        print(f"Epoch: {i}/{epoch_num}")
        tl, vl = 0.0, 0.0
        for j in matrices:
            print('Matrix', j)
            train_set.data_folder = os.path.join(f"{inpdir}/{j}")
            valid_set.data_folder = os.path.join(f"{inpdir}/{j}")
            A = torch.load(f"{train_set.data_folder}/A.pt").to_sparse_csr()
            image = torch.load(f"{train_set.data_folder}/flags.pt").view(1, N, N)
            training_loss_, validation_loss_, time_history_, grad_history_, update_history_ = train_(image, A, epoch_num_per_matrix, train_loader, valid_loader, model, optimizer, loss_fn)
            tl += np.sum(training_loss_)
            vl += np.sum(validation_loss_)
            grad_history.extend(grad_history_)
            update_history.extend(update_history_)
        train_loss.append(tl)
        valid_loss.append(vl)
        time_history.append(time.time() - start_time)
        saveData(model, optimizer, i+start_epoch, log, outdir, suffix, train_loss, valid_loss, time_history, grad_history, update_history)
    end_time = time.time()
    print("Took", end_time-start_time, 's')

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2)
    axes[0].plot(train_loss, label='training', c='blue')
    axes[1].plot(valid_loss, label='validation', c='orange')
    axes[0].legend()
    axes[1].legend()
    plt.savefig("train_loss.png")
    plt.clf()
    plt.plot(grad_history)
    plt.savefig("train_grad.png")
    plt.clf()
    plt.plot(update_history)
    plt.savefig("train_update.png")
