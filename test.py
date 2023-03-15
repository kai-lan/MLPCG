import os
import torch
from cg_tests import *
from model import *
from lib.read_data import *
from lib.discrete_laplacian import *
from torch.nn.functional import normalize
import time
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_default_dtype(torch.float32)
torch.set_grad_enabled(False) # disable autograd globally

def flip_data(rhs, flags):
    flipped_rhs = np.flip(rhs.reshape((N,)*DIM), axis=0).flatten()
    flipped_flags = np.flip(flags.reshape((N,)*DIM), axis=0).flatten()
    empty_cells = np.where(flipped_flags == 3)[0]
    n = N+2
    bd = box_bd(n, DIM)
    flipped_A = lap_with_bc(n, DIM, bd=bd, air=empty_cells, bd_padding=False, dtype=np.float32)
    return flipped_rhs, flipped_flags, flipped_A

# test in single or double precision
N = 64
DIM = 2
frame = 988 # 1 - 1000
norm_type = 'l2'

# train_matrices = set(np.load(f"{OUT_PATH}/output_2D_256/matrices_trained.npy"))
# frames = set(np.arange(1, 1001)) - train_matrices
# frame = list(frames)[2] # Randome frame
print("Testing frame", frame)
# large dambreak
dambreak_path = os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}") #_smoke include boundary
A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin")).astype(np.float32)
rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin")).astype(np.float32)
flags_sp = read_flags(os.path.join(dambreak_path, f"flags_{frame}.bin"))
# rhs_sp, flags_sp, A_sp = flip_data(rhs_sp, flags_sp)
A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32)
rhs = torch.tensor(rhs_sp, dtype=torch.float32)
flags = torch.tensor(flags_sp, dtype=torch.float32)


fluidnet_model_eigs_file = os.path.join(OUT_PATH, f"output_2D_64", f"model_dambreak.pth")
fluidnet_model_eigs = FluidNet()
fluidnet_model_eigs.load_state_dict(torch.load(fluidnet_model_eigs_file))


# fluidnet_model_res_file = os.path.join(DATA_PATH, f"output_single_{DIM}D_{N}", f"model_frame_800_res.pth")
# fluidnet_model_res = FluidNet()
# fluidnet_model_res.load_state_dict(torch.load(fluidnet_model_res_file))

verbose = False
max_iter = 1000
tol = 1e-5
t0 = time.time()
x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), max_iter, tol=tol, norm_type=norm_type, verbose=verbose)
print("CG took", time.time()-t0, 's')

def dcdm_predict(dcdm_model):
    global flags
    def predict(r):
        with torch.no_grad():
            r = normalize(r, dim=0)
            r = r.view(1, 1, N, N)
            x = dcdm_model(r).flatten()
        return x
    return predict

def fluidnet_predict(fluidnet_model):
    global flags
    def predict(r):
        with torch.no_grad():
            r = normalize(r, dim=0)
            b = torch.stack([r, flags]).view(1, 2, N, N)
            x = fluidnet_model(b).flatten()
        return x
    return predict

t0 = time.time()
x_fluidnet_eigs, res_fluidnet_eigs = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_eigs), max_iter, tol=tol, norm_type=norm_type, verbose=verbose)
print("FluidNet took", time.time()-t0, 's')
# x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_res), max_iter)

import matplotlib.pyplot as plt

plt.plot(res_fluidnet_eigs, label='fluidnet_eigs')
# plt.plot(res_fluidnet_res, label='fluidnet_res')
plt.plot(res_cg, label='cg')
if norm_type == 'l2': plt.yscale('log')
plt.title(f"{norm_type} VS Iterations")
plt.legend()
plt.savefig("test_loss.png")
