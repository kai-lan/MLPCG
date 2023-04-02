import os, sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cg_tests import *
from tqdm import tqdm
from model import *
from train import train_, saveData
from lib.read_data import *
from lib.dataset import MyDataset
from lib.write_log import LoggingWriter
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
    log = LoggingWriter()
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))
    return training_loss, validation_loss, time_history


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
    torch.set_default_dtype(torch.float32)
    N = 64
    DIM = 2
    dim2 = N**DIM
    lr = 0.001
    epoch_num = 10
    cuda = torch.device("cuda") # Use CUDA for training

    frame = 160
    num_rhs = 200
    b_size = 20

    data_path = os.path.join(DATA_PATH, f"dambreak_N{N}_{num_rhs}", f"preprocessed/{frame}")
    A = torch.load(f"{data_path}/A.pt")
    rhs = torch.tensor(torch.load(f"{data_path}/rhs.pt"))
    flags = torch.tensor(torch.load(f"{data_path}/flags.pt"))
    A = A.to(cuda)
    rhs, flags = rhs.to(cuda), flags.to(cuda)

    model = FluidNet(ks=3)
    model.move_to(cuda)
    loss_fn = model.residual_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_size = round(0.8 * num_rhs)
    perm = np.random.permutation(num_rhs)
    train_set = MyDataset(data_path, perm[:train_size], (2,)+(N,)*DIM)
    valid_set = MyDataset(data_path, perm[train_size:], (2,)+(N,)*DIM)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False)

    outdir = os.path.join(OUT_PATH, f"output_single_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    suffix = f"frame_{frame}_downsample"

    for_train = False
    for_test = True

    if for_train:
        training_loss, validation_loss, time_history = train_(A, epoch_num, train_loader, valid_loader, model, optimizer, loss_fn)
        saveData(model, optimizer, epoch_num, None, outdir, suffix, training_loss, validation_loss, time_history)

    def fluidnet_predict(fluidnet_model):
        global flags
        def predict(r):
            with torch.no_grad():
                r = nn.functional.normalize(r, dim=0)
                b = torch.stack([r, flags]).view(1, 2, N, N)
                x = fluidnet_model(b).flatten()
            return x
        return predict

    if for_test:
        checkpt = torch.load(f"{outdir}/checkpt_{suffix}.tar")
        model.load_state_dict(checkpt['model_state_dict'])
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        model.eval()
        x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model), max_it=200, tol=1e-4, verbose=True)

        print("Fluidnet", res_fluidnet_res[-1])

        # A = readA_sparse(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"A_{frame}.bin")).astype(np.float32)
        # rhs = load_vector(os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}", f"div_v_star_{frame}.bin")).astype(np.float32)
        # x, res_history = CG(rhs, A, np.zeros_like(rhs), max_it=1000, tol=1e-5, verbose=False)
        # r = rhs - A @ x
        # print(f"CG residual after {len(res_history)} iterations", np.linalg.norm(r)/np.linalg.norm(rhs))


    # fig, axes = plt.subplots(2)
    # axes[0].plot(training_loss, label='training')
    # axes[1].plot(validation_loss, label='validation')
    # plt.savefig("loss.png", bbox_inches='tight')
