import os, sys
sys.path.insert(1, 'lib')
os.environ['OMP_NUM_THREADS'] = '8'
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
from lib.global_clock import GlobalClock
from model import *
# from old_model import *
# from sm_model import *
from sm_model_3d import *
from loss_functions import residual_loss
import matplotlib.pyplot as plt

def move_data(data, device):
    if len(data) > 0:
        for i in range(len(data)):
            data[i] = data[i].to(device)
    else:
        data[:] = data.to(device)

def validation(train_loader, validation_loader, model, loss_fn, image, A, fluid_cells):
    tot_loss_train, tot_loss_val = 0, 0
    with torch.no_grad():
        for data in train_loader:
            # data = data.to(A.device)
            x_pred = model(image, data) # input: (bs, 1, dim, dim)

            tot_loss_train += loss_fn(x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells], data[:, 0].flatten(1)[:, fluid_cells], A, False)
        for data in validation_loader:
            # data = data.to(A.device)
            x_pred = model(image, data) # input: (bs, 1, dim, dim)
            tot_loss_val += loss_fn(x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells], data[:, 0].flatten(1)[:, fluid_cells], A, False)

    return tot_loss_train.item(), tot_loss_val.item()


def train_(image, A, fluid_cells, epoch_num, train_loader, validation_loader, model, optimizer, loss_fn):
    training_loss = []
    validation_loss = []
    time_history = []
    grad_history = []
    update_history = []
    t0 = time.time()

    tot_loss_train, tot_loss_val = validation(train_loader, validation_loader, model, loss_fn, image, A, fluid_cells)
    training_loss.append(tot_loss_train)
    validation_loss.append(tot_loss_val)
    time_history.append(time.time() - t0)
    print(training_loss[-1], validation_loss[-1], f"(0 / {epoch_num})")


    for i in range(1, epoch_num+1):
        for ii, data in enumerate(tqdm(train_loader), 1):
            # data = data.to(A.device)
            x_pred = model(image, data)
            x_pred = x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells] # (bs, 1, N, N) -> (bs, N, N) -> (bs, N*N) -> (bs, fluid_part)
            data = data.squeeze(dim=1).flatten(1)[:, fluid_cells]
            loss = loss_fn(x_pred, data, A)
            loss.backward()

            # if ii % 2 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            del loss, x_pred

        dLdw = 0.0
        tot_loss_train, tot_loss_val = validation(train_loader, validation_loader, model, loss_fn, image, A, fluid_cells)
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(training_loss[-1], validation_loss[-1], f"grad {dLdw}" f"({i} / {epoch_num})")

    return training_loss, validation_loss, time_history, grad_history, update_history

def saveData(model, optimizer, epoch, log, outdir, suffix, train_loss, valid_loss, time_history, grad_history, update_history, overwrite=True):
    if log is not None:
        log.record({"N": N,
                    "DIM": DIM,
                    "bc": bc,
                    "lr": lr,
                    "Epoches per matrix": epoch_num_per_matrix,
                    "Epoches": epoch,
                    "batch size": b_size,
                    "Num BCs": len(bcs),
                    "Num RHS": num_rhs})
        log.write(os.path.join(outdir, f"settings_{suffix}.log"), overwrite=overwrite)
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

## Each parameter can have its own learning rate, etc...
def create_param_groups_from_old_model(old_levels, new_model):
    slow_lr = lr*0.1
    param_groups = [{'params': new_model.l.parameters(), 'name':'l'}]
    for i in range(old_levels):
        param_groups.append({'params': new_model.l0[i].parameters(), 'name':f'l0.{i}', 'lr':slow_lr})
        param_groups.append({'params': new_model.l1[i].parameters(), 'name':f'l1.{i}', 'lr':slow_lr})
        param_groups.append({'params': new_model.c0[i].parameters(), 'name':f'c0.{i}', 'lr':slow_lr})
        param_groups.append({'params': new_model.c1[i].parameters(), 'name':f'c1.{i}', 'lr':slow_lr})
    for i in range(old_levels, new_model.n):
        param_groups.append({'params': new_model.l0[i].parameters(), 'name':f'l0.{i}'})
        param_groups.append({'params': new_model.l1[i].parameters(), 'name':f'l1.{i}'})
        param_groups.append({'params': new_model.c0[i].parameters(), 'name':f'c0.{i}'})
        param_groups.append({'params': new_model.c1[i].parameters(), 'name':f'c1.{i}'})
    return param_groups
def create_param_groups(model):
    param_groups = [{'params': model.l.parameters(), 'name':'l'}]
    for i in range(model.n):
        param_groups.append({'params': model.l0[i].parameters(), 'name':f'l0.{i}'})
        param_groups.append({'params': model.l1[i].parameters(), 'name':f'l1.{i}'})
        param_groups.append({'params': model.c0[i].parameters(), 'name':f'c0.{i}'})
        param_groups.append({'params': model.c1[i].parameters(), 'name':f'c1.{i}'})
    return param_groups

def transfer_weights_from_old_model(old_model_dir, old_model_suffix, new_model):
    ep, model_params, optim_params, train_loss, valid_loss, time_history, grad_history, update_history = loadData(old_model_dir, old_model_suffix)
    own_state = new_model.state_dict()
    for name, param in model_params.items():
        if name not in own_state: continue
        own_state[name].copy_(param)

if __name__ == '__main__':
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    N = 128
    DIM = 3
    lr = 0.0001
    epoch_num_per_matrix = 5
    epoch_num = 100
    epochs_per_save = 5
    shape = (1,)+(N,)*DIM
    bcs = [
        (f'dambreak_N{N}',                  (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        (f'dambreak_hill_N{N}_N{N*2}',     (N*2,)+(N,)*(DIM-1),   np.linspace(1, 200, 10, dtype=int)),
        (f'dambreak_dragons_N{N}_N{N*2}',  (N*2,)+(N,)*(DIM-1),    [1, 6, 10, 15, 21, 35, 44, 58, 81, 101, 162, 188]),
        (f'ball_cube_N{N}',                 (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'ball_bowl_N{N}',                 (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'standing_dipping_block_N{N}',    (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'standing_rotating_blade_N{N}',   (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        (f'waterflow_pool_N{N}',            (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'waterflow_panels_N{N}',          (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'waterflow_rotating_cube_N{N}',   (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'smoke_moving_donuts_N{N}',       (N,)*DIM,               np.linspace(10, 200, 10, dtype=int))
    ]
    bc = 'mixedBCs11'
    b_size = 128

    num_matrices = np.zeros(len(bcs), dtype=int)
    for i, bc in enumerate(bcs):
        num_matrices[i:] += len(bcs[i][-1])

    num_ritz = 1600
    num_rhs = 640+128 # number of ritz vectors for training for each matrix
    # num_rhs = 800
    kernel_size = 3 # kernel size
    num_imgs = 3

    if DIM == 2: num_levels = 6
    else: num_levels = 4

    cuda = torch.device("cuda") # Use CUDA for training

    resume = False
    randomize = True

    outdir = os.path.join(OUT_PATH, f"output_{DIM}D_{N}")

    suffix =  f'mixedBCs_M{len(bcs)}_ritz{num_ritz}_rhs{num_rhs}_l{num_levels}_test'

    os.makedirs(outdir, exist_ok=True)

    log = LoggingWriter()

    if DIM == 2:
        model = SmallSMModelDn(num_levels, num_imgs)
    else:
        # model = SPDSMModelDn3D(num_levels)
        model = SmallSMModelDn3D(num_levels, num_imgs, "trilinear")

    model.move_to(cuda)
    loss_fn = residual_loss

    # transfer_weights_from_old_model(outdir, f'mixedBCs_M{len(bcs)}_ritz{num_ritz}_rhs{num_rhs}_l3_trilinear', model)

    # optimizer = optim.Adam(create_param_groups_from_old_model(num_levels-1, model), lr=lr)
    # optimizer = optim.Adam(create_param_groups(model), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    if resume:
        ep, model_params, optim_params, train_loss, valid_loss, time_history, grad_history, update_history = loadData(outdir, suffix)
        model.load_state_dict(model_params)
        optimizer.load_state_dict(optim_params)
        start_epoch = len(train_loss)
    else:
        train_loss, valid_loss, time_history, grad_history, update_history = [], [], [], [], []
        start_epoch = 0


    train_size = 640
    perm = np.random.permutation(num_rhs)

    fluid_cells = None
    def transform(x):
        global fluid_cells, shape
        b = torch.zeros(np.prod(shape), dtype=torch.float32, device=cuda)
        b[fluid_cells] = x
        b = b.reshape(shape)
        return b

    train_set = MyDataset(None, perm[:train_size], transform)
    valid_set = MyDataset(None, perm[train_size:], transform)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False, num_workers=0)

    start_time = time.time()

    shuffled_matrices = range(num_matrices[-1])
    for i in range(start_epoch+1, epoch_num+1):
        tl, vl = 0.0, 0.0
        if randomize:
            shuffled_matrices = np.random.permutation(range(num_matrices[-1]))
        for j, mat in enumerate(shuffled_matrices):
            bc_id = np.searchsorted(num_matrices, mat, side='right')
            frame_id = (mat - num_matrices[bc_id-1])%len(shuffled_matrices)
            bc = bcs[bc_id][0]
            shape = (1,)+bcs[bc_id][1]
            frame = bcs[bc_id][2][frame_id]
            if DIM == 2: inpdir = f"{DATA_PATH}/{bc}_200/preprocessed"
            else:        inpdir = f"{DATA_PATH}/{bc}_200_{DIM}D/preprocessed"
            print(f"Epoch: {i}/{epoch_num}")
            print('BC:', bc, 'frame:', frame, f'progress: {j}/{len(shuffled_matrices)}')

            train_set.data_folder = os.path.join(f"{inpdir}/{frame}")
            valid_set.data_folder = os.path.join(f"{inpdir}/{frame}")

            A = torch.load(f"{train_set.data_folder}/A.pt", map_location='cuda')
            image = torch.load(f"{train_set.data_folder}/flags_binary_{num_imgs}.pt", map_location='cuda').view((num_imgs,)+shape[1:])

            fluid_cells = torch.load(f"{train_set.data_folder}/fluid_cells.pt", map_location='cuda')
            training_loss_, validation_loss_, time_history_, grad_history_, update_history_ = train_(image, A, fluid_cells, epoch_num_per_matrix, train_loader, valid_loader, model, optimizer, loss_fn)

            tl += np.sum(training_loss_)
            vl += np.sum(validation_loss_)
            grad_history.extend(grad_history_)
            update_history.extend(update_history_)
        train_loss.append(tl)
        valid_loss.append(vl)
        time_history.append(time.time() - start_time)

        saveData(model, optimizer, i, log, outdir, suffix, train_loss, valid_loss, time_history, grad_history, update_history, overwrite=(not resume))
        if i % epochs_per_save == 0:
            saveData(model, optimizer, i, log, outdir, suffix+f'_{i}', train_loss, valid_loss, time_history, grad_history, update_history, overwrite=(not resume))

    end_time = time.time()
    print("Took", end_time-start_time, 's')

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2)
    axes[0].plot(train_loss, label='training', c='blue')
    axes[1].plot(valid_loss, label='validation', c='orange')
    axes[0].legend()
    axes[1].legend()
    plt.savefig("train_loss.png")
    # plt.clf()
    # plt.plot(grad_history)
    # plt.savefig("train_grad.png")
    # plt.clf()
    # plt.plot(update_history)
    # plt.savefig("train_update.png")
