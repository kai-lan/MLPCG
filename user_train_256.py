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
from lib.write_log import LoggingWriter
from lib.dataset import *
from lib.GLOBAL_VARS import *
from lib.global_clock import GlobalClock
from loss_functions import residual_loss
from model import *
from sm_model import *
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        log: LoggingWriter,
        optimizer: optim.Optimizer,
        num_epoch_per_image: int,
        gpu_id: int,
        outdir: str,
        filename: str):

        self.log = log
        self.gpu_id = gpu_id
        self.num_epoch_per_image = num_epoch_per_image
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = residual_loss
        self.outdir = outdir
        self.filename = filename
        self.model = model


    def _validation(self, image, A, fluid_cells):
        tot_loss_train, tot_loss_val = 0, 0
        with torch.no_grad():
            for data in self.train_loader:
                data = data.to(A.device)
                x_pred = self.model(image, data)
                tot_loss_train += self.loss_fn(x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells], data[:, 0].flatten(1)[:, fluid_cells], A)
            for data in self.valid_loader:
                data = data.to(A.device)
                x_pred = self.model(image, data)
                tot_loss_val += self.loss_fn(x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells], data[:, 0].flatten(1)[:, fluid_cells], A)
        return tot_loss_train.item(), tot_loss_val.item()

    def train_per_image(self, image, A, fluid_cells):
        training_loss = []
        validation_loss = []
        time_history = []
        grad_history = []
        update_history = []
        t0 = time.time()

        tot_loss_train, tot_loss_val = self._validation(image, A, fluid_cells)
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time() - t0)
        print(f"[GPU {self.gpu_id}]", training_loss[-1], validation_loss[-1], f"(0 / {self.num_epoch_per_image})")

        for i in range(1, self.num_epoch_per_image+1):
            for ii, data in enumerate(self.train_loader, 1):
                data = data.to(A.device)
                x_pred = self.model(image, data)
                x_pred = x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells] # (bs, 1, N, N) -> (bs, N, N) -> (bs, N*N) -> (bs, fluid_part)
                data = data.squeeze(dim=1).flatten(1)[:, fluid_cells]
                loss = self.loss_fn(x_pred, data, A)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                del loss, x_pred

            tot_loss_train, tot_loss_val = self._validation(image, A, fluid_cells)
            training_loss.append(tot_loss_train)
            validation_loss.append(tot_loss_val)
            time_history.append(time.time() - t0)
            print(f"[GPU {self.gpu_id}]", training_loss[-1], validation_loss[-1], f"({i} / {self.num_epoch_per_image})")

        if self.gpu_id == 0:
            return training_loss, validation_loss, time_history, grad_history, update_history
        return None, None, None, None, None

    def saveData(self, epoch, train_loss, valid_loss, time_history, grad_history, update_history, overwrite=True, suffix=''):
        if self.log is not None:
            self.log.record({"N": N,
                        "DIM": DIM,
                        "lr": lr,
                        "Epoches per matrix": epoch_num_per_matrix,
                        "Epoches": epoch,
                        "batch size": self.train_loader.batch_size,
                        "Num matrices": total_matrices,
                        "Num RHS": num_rhs})
            self.log.write(os.path.join(self.outdir, f"settings_{self.filename}.log"), overwrite=overwrite)
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_loss': train_loss,
                'validation_loss': valid_loss,
                'time': time_history,
                'grad': grad_history,
                'update': update_history
                }, os.path.join(self.outdir, f"checkpt_{self.filename}{suffix}.tar"))

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



N = 256
DIM = 3
bcs = [
        (f'dambreak_N{N}',                  (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        (f'dambreak_hill_N{N//2}_N{N}',     (N,)+(N//2,)*(DIM-1),   np.linspace(1, 200, 10, dtype=int)),
        (f'dambreak_dragons_N{N//2}_N{N}',  (N,)+(N//2,)*(DIM-1),    [1, 6, 10, 15, 21, 35, 44, 58, 81, 101, 162, 188]),
        (f'ball_cube_N{N}',                 (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'ball_bowl_N{N}',                 (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'standing_dipping_block_N{N}',    (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'standing_rotating_blade_N{N}',   (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        (f'waterflow_pool_N{N}',            (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        (f'waterflow_panels_N{N}',          (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        (f'waterflow_rotating_cube_N{N}',   (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:])
    ]
lr = 0.0001
epoch_num_per_matrix = 5
epoch_num = 100
total_matrices = np.sum([len(bc[-1]) for bc in bcs]) # number of matrices chosen for training
num_ritz = 1600
num_rhs = 800 # number of ritz vectors for training for each matrix

def main(rank, world_size, batch_size):
    print('GPU ID', rank)
    ddp_setup(rank, world_size)

    resume = True
    randomize = True


    epochs_per_save = 5

    kernel_size = 3 # kernel size
    num_imgs = 3
    num_levels = 3 # depth of the network

    filename =  f'mixedBCs_M{total_matrices}_ritz{num_ritz}_rhs{num_rhs}_imgs{num_imgs}_lr0.0001_from128'
    outdir = os.path.join(OUT_PATH, f"output_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)

    log = LoggingWriter()

    if DIM == 2: model = SmallSMModelDn(n=6, num_imgs=num_imgs)
    else:        model = SmallSMModelDn3D(n=num_levels, num_imgs=num_imgs)

    model = model.to(rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume:
        ep, model_params, optim_params, train_loss, valid_loss, time_history, grad_history, update_history = loadData(outdir, filename)
        model.load_state_dict(model_params)
        optimizer.load_state_dict(optim_params)
        start_epoch = len(train_loss)
    else:
        train_loss, valid_loss, time_history, grad_history, update_history = [], [], [], [], []
        start_epoch = 0

    model = DDP(model, device_ids=[rank])

    train_size = round(0.8 * num_rhs)
    perm = np.random.permutation(num_rhs)

    train_set = DDPDataset(perm[:train_size])
    valid_set = DDPDataset(perm[train_size:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, sampler=DistributedSampler(train_set))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, sampler=DistributedSampler(valid_set))


    trainer = Trainer(model, train_loader, valid_loader, log, optimizer, epoch_num_per_matrix, rank, outdir, filename)
    start_time = time.time()

    for i in range(start_epoch+1, epoch_num+start_epoch+1):
        tl, vl = 0.0, 0.0
        if randomize: np.random.shuffle(bcs)
        for count, (bc, sha, matrices) in enumerate(bcs, 1):
            shape = (1,)+sha
            num_matrices = len(matrices)
            if randomize: np.random.shuffle(matrices)
            for j_mat, j in enumerate(matrices, 1):
                print(f"[GPU {rank}]", f"Epoch: {i}/{epoch_num}")
                print(f"[GPU {rank}]", bc, f'{count}/{len(bcs)}')
                print(f"[GPU {rank}]", 'Matrix', j, f'{j_mat}/{num_matrices}')

                valid_set.set_image(bc, j, shape)
                A, image, fluid_cells = train_set.set_and_return_image(bc, j, shape)
                A = A.to(rank)
                image = image.view((num_imgs,)+sha).to(rank)
                fluid_cells = fluid_cells.to(rank)
                training_loss_, validation_loss_, time_history_, grad_history_, update_history_ \
                    = trainer.train_per_image(image, A, fluid_cells)

                if rank == 0:
                    tl += np.sum(training_loss_)
                    vl += np.sum(validation_loss_)
                    grad_history.extend(grad_history_)
                    update_history.extend(update_history_)
        if rank == 0:
            train_loss.append(tl)
            valid_loss.append(vl)
            time_history.append(time.time() - start_time)
            trainer.saveData(i, train_loss, valid_loss, time_history, grad_history, update_history, overwrite=(not resume))
            if i % 5 == 0:
                trainer.saveData(i, train_loss, valid_loss, time_history, grad_history, update_history, overwrite=(not resume), suffix=f'_{i}')

    end_time = time.time()
    print("Took", end_time-start_time, 's')

    destroy_process_group()

if __name__ == '__main__':
    # batch size | GPU usage (MB)
    #      16    |  15596
    #      32    |  24116
    #      64    |  46440

    batch_size = 64
    num_gpus = 1

    mp.spawn(main, args=(num_gpus, batch_size), nprocs=num_gpus)













