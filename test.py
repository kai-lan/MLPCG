import os
import torch
from cg_tests import *
from model import *
from lib.read_data import *
from lib.discrete_laplacian import *
from torch.nn.functional import normalize

def flip_data(rhs, flags):
    flipped_rhs = np.flip(rhs.reshape((N,)*DIM), axis=0).flatten()
    flipped_flags = np.flip(flags.reshape((N,)*DIM), axis=0).flatten()
    empty_cells = np.where(flipped_flags == 3)[0]
    n = N+2
    bd = box_bd(n, DIM)
    flipped_A = lap_with_bc(n, DIM, bd=bd, air=empty_cells, bd_padding=False, dtype=np.float32)
    return flipped_rhs, flipped_flags, flipped_A

# test in single or double precision
torch.set_default_dtype(torch.float32)
N = 64
DIM = 2
frame = 5 # 1 - 1000
include_bd = False
prefix = "_" if include_bd else ''
norm_type = 'l2'
# Dambreak
# dambreak_path = os.path.join(DATA_PATH, f"{prefix}largedambreak_{DIM}D_{N}")
# A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin")).astype(np.float32)
# rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin")).astype(np.float32)
# A = torch.load(os.path.join(dambreak_path, f"preprocessed/{frame}", f"A.pt"))
# rhs = torch.load(os.path.join(dambreak_path, f"preprocessed/{frame}", f"rhs.pt"))
# flags = torch.load(os.path.join(dambreak_path, f"preprocessed/{frame}", f"flags.pt"))
# flipped_rhs = rhs.reshape((N,)*DIM).flip(dims=(0,)).faltten()
# flipped_flags = flags.reshape((N,)*DIM).flip(dims=(0,)).faltten()

# large dambreak
largedambreak_path = os.path.join(DATA_PATH, f"{prefix}dambreak_rightbottom_{DIM}D_{N}") #_smoke include boundary
A_sp = readA_sparse(os.path.join(largedambreak_path, f"A_{frame}.bin")).astype(np.float32)
rhs_sp = load_vector(os.path.join(largedambreak_path, f"div_v_star_{frame}.bin")).astype(np.float32)
flags_sp = read_flags(os.path.join(largedambreak_path, f"flags_{frame}.bin"))
# rhs_sp, flags_sp, A_sp = flip_data(rhs_sp, flags_sp)
A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32)
rhs = torch.from_numpy(rhs_sp)
flags = torch.tensor(flags_sp, dtype=torch.float32)

# Smoke data
# smoke_path = os.path.join(DATA_PATH, f"{prefix}smoke_{DIM}D_{N}") #_smoke include boundary
# A_sp = readA_sparse(os.path.join(smoke_path, f"A_{frame}.bin"))
# rhs_sp = load_vector(os.path.join(smoke_path, f"div_v_star_{frame}.bin"))
# A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32)
# rhs = torch.from_numpy(rhs_sp).float()
# flags = torch.tensor(read_flags(os.path.join(smoke_path, f"flags_{frame}.bin")), dtype=torch.float32)
# print(A.shape, rhs.shape, flags.shape)

# Empty box data
# emptybox_path = os.path.join(dcdm_data_path, f"train_{DIM}D_{N}")
# A_sp = readA_sparse(os.path.join(emptybox_path, f"A_solid.bin"), 'f')
# rhs_sp = np.load(os.path.join(emptybox_path, "b_solid_100.npy"))
# A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32)
# rhs = torch.from_numpy(rhs_sp).float()
# flags = torch.ones((N,)*DIM) * 2
# flags[0] = 0
# flags[-1] = 0
# flags[:, 0] = 0
# flags[:, -1] = 0
# flags = flags.flatten()

fluidnet_model_eigs_file = os.path.join(OUT_PATH, f"{prefix}output_{DIM}D_{N}", f"{prefix}model_dambreak.pth")
fluidnet_model_eigs = FluidNet()
fluidnet_model_eigs.load_state_dict(torch.load(fluidnet_model_eigs_file))

# fluidnet_model_cano_file = os.path.join(DATA_PATH, f"{prefix}output_single_{DIM}D_{N}", f"{prefix}model_frame_800_cano.pth")
# fluidnet_model_cano = FluidNet()
# fluidnet_model_cano.load_state_dict(torch.load(fluidnet_model_cano_file))

# fluidnet_model_res_file = os.path.join(DATA_PATH, f"{prefix}output_single_{DIM}D_{N}", f"{prefix}model_frame_800_res.pth")
# fluidnet_model_res = FluidNet()
# fluidnet_model_res.load_state_dict(torch.load(fluidnet_model_res_file))

# fluidnet_model_empty_file = os.path.join(DATA_PATH, f"{prefix}output_{DIM}D_{N}", f"{prefix}model_empty.pth")
# fluidnet_model_empty = FluidNet()
# fluidnet_model_empty.load_state_dict(torch.load(fluidnet_model_empty_file))

# dcdm_model_file = os.path.join(dcdm_data_path, f"{prefix}output_{DIM}D_{N}", f"{prefix}model_empty.pth")
# dcdm_model = DCDM(DIM)
# dcdm_model.load_state_dict(torch.load(dcdm_model_file))

max_iter = 50

x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), max_iter, norm_type=norm_type)

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

# x_dcdm, res_dcdm = dcdm(rhs, A, torch.zeros_like(rhs), dcdm_predict(dcdm_model), max_iter)
x_fluidnet_eigs, res_fluidnet_eigs = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_eigs), max_iter, tol=1e-14, norm_type=norm_type)
# x_fluidnet_cano, res_fluidnet_cano = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_cano), max_iter)
# x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_res), max_iter)
# x_fluidnet_empty, res_fluidnet_empty = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_empty), max_iter)

import matplotlib.pyplot as plt
# plt.autoscale()
# plt.plot(res_dcdm, label='dcdm_empty')
plt.plot(res_fluidnet_eigs, label='fluidnet_eigs')
# plt.plot(res_fluidnet_cano, label='fluidnet_cano')
# plt.plot(res_fluidnet_res, label='fluidnet_res')
# plt.plot(res_fluidnet_empty, label='fluidnet_empty')
plt.plot(res_cg, label='cg')
if norm_type == 'l2': plt.yscale('log')
plt.title(f"{norm_type} VS Iterations")
plt.legend()
plt.savefig("test_loss.png")
