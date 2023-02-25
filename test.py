import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dcdm_data_path = os.path.join(dir_path, "data_dcdm")
fluidnet_data_path = os.path.join(dir_path, "data_fluidnet")
import torch
from cg_tests import *
from model import *
from lib.read_data import *
from torch.nn.functional import normalize

N = 64
DIM = 2
frame = 800 # 1 - 1000
include_bd = False
prefix = "_" if include_bd else ''

# Dambreak
dambreak_path = os.path.join(fluidnet_data_path, f"{prefix}dambreak_{DIM}D_{N}")
A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin"))
rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin"))
A = torch.load(os.path.join(dambreak_path, "preprocessed", f"A_{frame}.pt"))
x = torch.load(os.path.join(dambreak_path, "preprocessed", f"x_{frame}.pt"))
rhs, flags = x[0], x[1]

# Smoke data
# smoke_path = os.path.join(fluidnet_data_path, f"{prefix}smoke_{DIM}D_{N}") #_smoke include boundary
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

fluidnet_model_eigs_file = os.path.join(fluidnet_data_path, f"{prefix}output_single_{DIM}D_{N}", f"{prefix}model_frame_800_eigs.pth")
fluidnet_model_eigs = FluidNet()
fluidnet_model_eigs.load_state_dict(torch.load(fluidnet_model_eigs_file))

fluidnet_model_cano_file = os.path.join(fluidnet_data_path, f"{prefix}output_single_{DIM}D_{N}", f"{prefix}model_frame_800_cano.pth")
fluidnet_model_cano = FluidNet()
fluidnet_model_cano.load_state_dict(torch.load(fluidnet_model_cano_file))

fluidnet_model_res_file = os.path.join(fluidnet_data_path, f"{prefix}output_single_{DIM}D_{N}", f"{prefix}model_frame_800_res.pth")
fluidnet_model_res = FluidNet()
fluidnet_model_res.load_state_dict(torch.load(fluidnet_model_res_file))

fluidnet_model_empty_file = os.path.join(fluidnet_data_path, f"{prefix}output_{DIM}D_{N}", f"{prefix}model_empty.pth")
fluidnet_model_empty = FluidNet()
fluidnet_model_empty.load_state_dict(torch.load(fluidnet_model_empty_file))

dcdm_model_file = os.path.join(dcdm_data_path, f"{prefix}output_{DIM}D_{N}", f"{prefix}model_empty.pth")
dcdm_model = DCDM(DIM)
dcdm_model.load_state_dict(torch.load(dcdm_model_file))

max_iter = 100
x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), max_iter)

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

x_dcdm, res_dcdm = dcdm(rhs, A, torch.zeros_like(rhs), dcdm_predict(dcdm_model), max_iter)
x_fluidnet_eigs, res_fluidnet_eigs = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_eigs), max_iter)
x_fluidnet_cano, res_fluidnet_cano = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_cano), max_iter)
x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_res), max_iter)
x_fluidnet_empty, res_fluidnet_empty = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(fluidnet_model_empty), max_iter)

import matplotlib.pyplot as plt
plt.plot(res_dcdm, label='dcdm_empty')
plt.plot(res_fluidnet_eigs, label='fluidnet_eigs')
plt.plot(res_fluidnet_cano, label='fluidnet_cano')
plt.plot(res_fluidnet_res, label='fluidnet_res')
plt.plot(res_fluidnet_empty, label='fluidnet_empty')
plt.plot(res_cg, label='cg')
plt.yscale('log')
plt.legend()
plt.savefig("test_loss.png")
