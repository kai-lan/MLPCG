import os, sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.benchmark as torch_benchmark
from cg_tests import *
from tqdm import tqdm
from model import *
from train import train_, saveData
from lib.read_data import *
from lib.dataset import MyDataset
from lib.write_log import LoggingWriter
import time, timeit
dir_path = os.path.realpath(__file__)
sys.path.insert(1, os.path.join(dir_path, 'lib'))


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

    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))
    return training_loss, validation_loss, time_history
class RitzDataset(Dataset):
    def __init__(self, data, flags, perm, shape):
        self.data = data
        self.flags = flags
        self.perm = perm
        self.shape = shape
        self.fluidcells = torch.where(flags == 2)[0]
    def __getitem__(self, index):
        index = self.perm[index]
        b = torch.zeros(N**DIM, dtype=torch.float32)
        b[self.fluidcells] = torch.from_numpy(self.data[index])
        x = torch.stack([
            b,
            self.flags
        ]).reshape(self.shape)
        return x
    def __len__(self):
        return len(self.perm)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
    torch.set_default_dtype(torch.float32)
    N = 128
    DIM = 2
    dim2 = N**DIM
    lr = 0.001
    epoch_num = 100
    cuda = torch.device("cuda") # Use CUDA for training

    image_type = 'flags'
    frame = 100 # 100, 25, 10
    num_ritz = 200
    num_rhs = 200
    b_size = 20
    denoised = False

    data_path = os.path.join(DATA_PATH, f"dambreak_N{N}_200", f"preprocessed/{frame}")
    A = torch.load(f"{data_path}/A.pt")
    rhs = torch.tensor(torch.load(f"{data_path}/rhs.pt"))
    # rhs = torch.tensor(torch.load(f"{data_path}/rhs_denoised.pt"))
    image = torch.tensor(torch.load(f"{data_path}/{image_type}.pt"))
    fluids = torch.argwhere(image == 2)
    if denoised:
        image_denoised = torch.tensor(torch.load(f"{data_path}/{image_type}_denoised.pt"))
        fluids_denoised = torch.argwhere(image_denoised == 2)

    A = A.to(cuda)

    model = FluidNet(ks=3)
    # model = NewModel1()
    model.move_to(cuda)
    loss_fn = model.residual_loss
    # loss_fn = model.energy_loss
    # loss_fn = model.scaled_loss_2
    # loss_fn = model.scaled_loss_A

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_size = round(0.8 * num_rhs)
    perm = np.random.permutation(num_rhs)
    def transform(x):
        x = x.reshape((2,)+(N,)*DIM)
        return x
    train_set = MyDataset(data_path, perm[:train_size], transform, denoised=denoised)
    valid_set = MyDataset(data_path, perm[train_size:], transform, denoised=denoised)

    # data = np.memmap(f"{data_path}/ritz_{num_ritz}.dat", dtype=np.float32, mode='r').reshape(num_ritz, len(torch.argwhere(image==2)))
    # train_set = RitzDataset(data, image, perm[:train_size], (2,)+(N,)*DIM)
    # valid_set = RitzDataset(data, image, perm[train_size:], (2,)+(N,)*DIM)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False)

    outdir = os.path.join(OUT_PATH, f"output_single_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    suffix = f"frame_{frame}_{image_type}"

    for_train = True
    for_test = True

    if for_train:
        training_loss, validation_loss, time_history = train_(A, epoch_num, train_loader, valid_loader, model, optimizer, loss_fn)
        saveData(model, optimizer, epoch_num, None, outdir, suffix, training_loss, validation_loss, time_history)
        fig, axes = plt.subplots(2)
        axes[0].plot(training_loss, label='training')
        axes[1].plot(validation_loss, label='validation')
        plt.savefig("loss.png", bbox_inches='tight')
    if for_test:
        checkpt = torch.load(f"{outdir}/checkpt_{suffix}.tar")
        model.load_state_dict(checkpt['model_state_dict'])
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        model.eval()
        image = image.to(cuda)
        if denoised: image_denoised = image_denoised.to(cuda)
        rhs = rhs.to(cuda)

        def fluidnet_predict(fluidnet_model):
            if denoised:
                # fluids_full = torch.where(abs(x_in[:, 1:] - 2) > 1e-12)
                def predict(r):
                    with torch.no_grad():
                        r = nn.functional.normalize(r, dim=0)
                        rr = torch.zeros_like(r)
                        rr[fluids_denoised] = r[fluids_denoised]
                        b = torch.stack([rr, image_denoised]).view(1, 2, N, N)
                        x = r.clone()
                        x[fluids_denoised] = fluidnet_model(b).flatten()[fluids_denoised]
                    return x
            else:
                def predict(r):
                    with torch.no_grad():
                        r = nn.functional.normalize(r, dim=0)
                        b = torch.stack([r, image]).view(1, 2, N, N)
                        x = fluidnet_model(b).flatten()
                    return x
            return predict

        # def torch_timer(loss):
        #     return torch_benchmark.Timer(
        #     stmt=f'torch_benchmark_dcdm_{loss}(model, image)',
        #     setup=f'from __main__ import torch_benchmark_dcdm_{loss}',
        #     globals={'model':model, 'image': image})

        # def torch_benchmark_dcdm_res(model, image):
        #     x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model, image), max_it=100, tol=1e-4, verbose=True)
        # timer_res = torch_timer('res')
        # result_res = timer_res.timeit(1)
        # print("FluidNet residual took", result_res.times[0], 's')

        x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model), max_it=100, tol=1e-4, verbose=True)
        print("Fluidnet", res_fluidnet_res[-1])

        A = readA_sparse(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"A_{frame}.bin")).astype(np.float32)
        rhs = load_vector(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"div_v_star_{frame}.bin")).astype(np.float32)
        t0 = timeit.default_timer()
        x, res_history = CG(rhs, A, np.zeros_like(rhs), max_it=1000, tol=1e-4, verbose=False)
        # r = rhs - A @ x
        print("CG took", timeit.default_timer()-t0, 's after', len(res_history), 'iterations', res_history[-1])

