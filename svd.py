import numpy as np
from model import *
from lib.read_data import *
from tqdm import tqdm
import scipy as sp
import matplotlib.pyplot as plt
torch.set_grad_enabled(False) # disable autograd globally

N = 64
DIM = 2
n = N**DIM
frame = 800
if 1:
    A = readA_sparse(f"data_fluidnet/dambreak_{DIM}D_{N}/A_{frame}.bin")
    flags = read_flags(f"data_fluidnet/dambreak_{DIM}D_{N}/flags_{frame}.bin")
    fluid_cells = np.where(flags == 2)[0]
    flags = torch.from_numpy(flags)
    num_fluid_cells = len(fluid_cells)
    print(num_fluid_cells)
    file = f"data_fluidnet/output_single_{DIM}D_{N}/model_frame_{frame}_eigs.pth"
    model = FluidNet()
    model.load_state_dict(torch.load(file))
    A_inv = np.zeros((num_fluid_cells, num_fluid_cells))
    for i, cell in enumerate(tqdm(fluid_cells)):
        rhs = torch.zeros(n)
        rhs[cell] = 1.0
        x = model(torch.stack([rhs, flags]).reshape((1, 2,)+(N,)*DIM)).flatten()
        x = x[fluid_cells]
        A_inv[:, i] = x.numpy()
    A = A[fluid_cells][:, fluid_cells]
else:
    A = readA_sparse(f"data_dcdm/train_{DIM}D_{N}/A_empty.bin", 'f')
    flags = torch.ones(n) * 2
    file = f"data_fluidnet/output_{DIM}D_{N}/model_empty.pth"
    model = FluidNet()
    model.load_state_dict(torch.load(file))
    A_inv = np.zeros((n, n))
    for i in tqdm(range(n)):
        rhs = torch.zeros(n)
        rhs[i] = 1.0
        x = model(torch.stack([rhs, flags]).reshape((1, 2,)+(N,)*DIM)).flatten()
        A_inv[:, i] = x.numpy()

if 0:
    I_approx = A @ A_inv
    err = np.identity(A.shape[0]) - I_approx
    print(err)
    plt.imshow(err, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig("img.png")
else:
    eigs = sp.linalg.eigvals(A.toarray())
    eigs = 1 / eigs
    inv_eigs = np.linalg.eigvals(A_inv)
    print(eigs)
    plt.scatter(eigs.real, eigs.imag, s=3)
    # plt.yscale("log")
    # err = eigs - inv_eigs
    # plt.imshow(err.reshape((N,)*DIM), origin='lower', cmap='jet')
    plt.savefig("svd.png")
