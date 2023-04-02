import os
import torch
from cg_tests import *
import cupyx.scipy.sparse as cpsp
from cupyx.profiler import benchmark as cuda_benchmark
import torch.utils.benchmark as torch_benchmark
from model import *
from lib.read_data import *
from lib.discrete_laplacian import *
from torch.nn.functional import normalize
import time, timeit
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

N = 256
DIM = 2
norm_type = 'l2'

train_matrices = set(np.load(f"{OUT_PATH}/output_2D_256/matrices_trained_50.npy"))
frames = set(np.arange(1, 1001)) - train_matrices
frame = list(frames)[138] # Random frame
# frame = list(train_matrices)[40]
# frame = 164
scene = 'dambreak'
print("Testing frame", frame, "scene", scene)

# dambreak
dambreak_path = os.path.join(DATA_PATH, f"{scene}_N{N}_200") #_smoke include boundary
A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin")).astype(np.float32)
rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin")).astype(np.float32)
flags_sp = read_flags(os.path.join(dambreak_path, f"flags_{frame}.bin"))
# rhs_sp, flags_sp, A_sp = flip_data(rhs_sp, flags_sp)

device = torch.device('cuda')
A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32, device=device)
rhs = torch.tensor(rhs_sp, dtype=torch.float32, device=device)
flags = torch.tensor(flags_sp, dtype=torch.float32, device=device)

NN = 64
num_mat = 100
num_ritz = 100
num_rhs = 100
fluidnet_model_res_file = os.path.join(OUT_PATH, f"output_2D_{NN}", f"model_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res.pth")
fluidnet_model_res = FluidNet()
fluidnet_model_res.move_to(device)
fluidnet_model_res.load_state_dict(torch.load(fluidnet_model_res_file))
fluidnet_model_res.eval()

fluidnet_model_eng_file = os.path.join(OUT_PATH, f"output_2D_{NN}", f"model_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_eng.pth")
fluidnet_model_eng = FluidNet()
fluidnet_model_eng.move_to(device)
fluidnet_model_eng.load_state_dict(torch.load(fluidnet_model_eng_file))
fluidnet_model_eng.eval()

fluidnet_model_scaled2_file = os.path.join(OUT_PATH, f"output_2D_{NN}", f"model_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_scaled2.pth")
fluidnet_model_scaled2 = FluidNet()
fluidnet_model_scaled2.move_to(device)
fluidnet_model_scaled2.load_state_dict(torch.load(fluidnet_model_scaled2_file))
fluidnet_model_scaled2.eval()

fluidnet_model_scaledA_file = os.path.join(OUT_PATH, f"output_2D_{NN}", f"model_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_scaledA.pth")
fluidnet_model_scaledA = FluidNet()
fluidnet_model_scaledA.move_to(device)
fluidnet_model_scaledA.load_state_dict(torch.load(fluidnet_model_scaledA_file))
fluidnet_model_scaledA.eval()

################
# CG
################
verbose = False
max_iter = 500
tol = 1e-4
atol = 1e-10 # safe guard for small rhs
t0 = timeit.default_timer()
x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
print("CG took", timeit.default_timer()-t0, 's after', len(res_cg), 'iterations')

###############
# CG GPU
###############
rhs_cp, A_cp = cp.array(rhs_sp, dtype=np.float32), cpsp.csr_matrix(A_sp, dtype=np.float32)
x_cg_cp, res_cg_cp = None, None
def cuda_benchmark_cg():
    global x_cg_cp, res_cg_cp
    x_cg_cp, res_cg_cp = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
result = cuda_benchmark(cuda_benchmark_cg, n_repeat=1)
print("CUDA CG took", result.gpu_times[0][0], 's after', len(res_cg_cp), 'iterations')


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


x_fluidnet_res, res_fluidnet_res = None, None
x_fluidnet_eng, res_fluidnet_eng = None, None
x_fluidnet_scale2, res_fluidnet_scale2 = None, None
x_fluidnet_scaleA, res_fluidnet_scaleA = None, None

def torch_benchmark_dcdm_res(model):
    global x_fluidnet_res, res_fluidnet_res
    x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model), max_iter, tol=tol, atol=atol)
def torch_benchmark_dcdm_eng(model):
    global x_fluidnet_eng, res_fluidnet_eng
    x_fluidnet_eng, res_fluidnet_eng = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model), max_iter, tol=tol, atol=atol)
def torch_benchmark_dcdm_scaled2(model):
    global x_fluidnet_scale2, res_fluidnet_scale2
    x_fluidnet_scale2, res_fluidnet_scale2 = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model), max_iter, tol=tol, atol=atol)
def torch_benchmark_dcdm_scaledA(model):
    global x_fluidnet_scaleA, res_fluidnet_scaleA
    x_fluidnet_scaleA, res_fluidnet_scaleA = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model), max_iter, tol=tol, atol=atol)
def torch_timer(loss):
    return torch_benchmark.Timer(
    stmt=f'torch_benchmark_dcdm_{loss}(model)',
    setup=f'from __main__ import torch_benchmark_dcdm_{loss}',
    globals={'model':eval(f"fluidnet_model_{loss}")})

timer_res = torch_timer('res')
result_res = timer_res.timeit(1)
print("FluidNet residual took", result_res.times[0], 's after', len(res_fluidnet_res), 'iterations')

timer_eng = torch_timer('eng')
result_eng = timer_eng.timeit(1)
print("FluidNet energy took", result_eng.times[0], 's after', len(res_fluidnet_eng), 'iterations')

timer_scaled2 = torch_timer('scaled2')
result_scaled2 = timer_scaled2.timeit(1)
print("FluidNet scaled2 took", result_scaled2.times[0], 's after', len(res_fluidnet_scale2), 'iterations')

timer_scaledA = torch_timer('scaledA')
result_scaledA = timer_scaledA.timeit(1)
print("FluidNet scaledA took", result_scaledA.times[0], 's after', len(res_fluidnet_scaleA), 'iterations')

import matplotlib.pyplot as plt

plt.plot(res_fluidnet_res, label='fluidnet_res')
plt.plot(res_fluidnet_eng, label='fluidnet_energy')
plt.plot(res_fluidnet_scale2, label='fluidnet_scale2')
plt.plot(res_fluidnet_scaleA, label='fluidnet_scaleA')
plt.plot(res_cg, label='cg')
if norm_type == 'l2': plt.yscale('log')
plt.title(f"{norm_type} VS Iterations")
plt.legend()
plt.savefig("test_loss.png")
