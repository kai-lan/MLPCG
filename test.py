import os
import torch
from cg_tests import *
import cupyx.scipy.sparse as cpsp
from cupyx.profiler import benchmark as cuda_benchmark
import torch.utils.benchmark as torch_benchmark
from model import *
from sm_model import *
from lib.read_data import *
from lib.discrete_laplacian import *
from torch.nn.functional import normalize
import time, timeit
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_grad_enabled(False) # disable autograd globally

def flip_data(rhs, flags):
    flipped_rhs = np.flip(rhs.reshape((N,)*DIM), axis=0).flatten()
    flipped_flags = np.flip(flags.reshape((N,)*DIM), axis=0).flatten()
    empty_cells = np.where(flipped_flags == 3)[0]
    n = N+2
    bd = box_bd(n, DIM)
    flipped_A = lap_with_bc(n, DIM, bd=bd, air=empty_cells, bd_padding=False, dtype=np.float32)
    return flipped_rhs, flipped_flags, flipped_A

pcg_precision = torch.float32
torch.set_default_dtype(pcg_precision)

N = 1024
DIM = 2
norm_type = 'l2'

# train_matrices = set(np.load(f"{OUT_PATH}/output_{DIM}D_{N}/matrices_trained_50.npy"))
# frames = set(np.arange(1, 151)) - train_matrices
# frame = list(frames)[10] # Random frame
# frame = list(train_matrices)[49]
frame = 1
if DIM == 2:
    scene = f'dambreak_N{N}_200'
else:
    scene = f'dambreak_N{N}_200_{DIM}D'

print("Testing frame", frame, "scene", scene)

# dambreak
dambreak_path = os.path.join(DATA_PATH, f"{scene}") #_smoke include boundary
A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin")).astype(np.float64)
rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin")).astype(np.float64)
flags_sp = read_flags(os.path.join(dambreak_path, f"flags_{frame}.bin"))
# ppc_sp = read_ppc(f"{dambreak_path}/active_cells_{frame}.bin", f"{dambreak_path}/ppc_{frame}.bin", N, DIM)
# levelset_sp = load_vector(f"{dambreak_path}/levelset_{frame}.bin")

device = torch.device('cuda')
A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=pcg_precision, device=device)
rhs = torch.tensor(rhs_sp, dtype=pcg_precision, device=device)
flags = torch.tensor(flags_sp, dtype=pcg_precision, device=device)
# ppc = torch.tensor(ppc_sp, dtype=pcg_precision, device=device)
# levelset = torch.tensor(levelset_sp, dtype=pcg_precision, device=device)

NN = 256
num_mat = 50
num_ritz = 800
num_rhs = 400

fluidnet_model_res_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_flags_smmodeld4.tar")
fluidnet_model_res = SmallSMModelDn3D(4) if DIM == 3 else SmallSMModelDn(4)
fluidnet_model_res.move_to(device)
fluidnet_model_res.load_state_dict(torch.load(fluidnet_model_res_file)['model_state_dict'])
fluidnet_model_res.eval()

# fluidnet_model_eng_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_flags_smmodel.tar")
# fluidnet_model_eng = SmallSMModel()
# fluidnet_model_eng.move_to(device)
# fluidnet_model_eng.load_state_dict(torch.load(fluidnet_model_eng_file)['model_state_dict'])
# fluidnet_model_eng.eval()

# fluidnet_model_scaled2_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_flags_fluidnet.tar")
# fluidnet_model_scaled2 = FluidNet()
# fluidnet_model_scaled2.move_to(device)
# fluidnet_model_scaled2.load_state_dict(torch.load(fluidnet_model_scaled2_file)['model_state_dict'])
# fluidnet_model_scaled2.eval()

# fluidnet_model_scaledA_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_dambreak_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_flags_smmodeld4.tar")
# fluidnet_model_scaledA = SmallSMModelDn(4)
# fluidnet_model_scaledA.move_to(device)
# fluidnet_model_scaledA.load_state_dict(torch.load(fluidnet_model_scaledA_file)['model_state_dict'])
# fluidnet_model_scaledA.eval()

verbose = False
cg_max_iter = 1500
pcg_max_iter = 200
tol = 1e-4
atol = 1e-10 # safe guard for small rhs
###############
# AMGCL
###############
t0 = timeit.default_timer()
x_amgcg, res_amgcg = AMGCG(rhs_sp, A_sp, np.zeros_like(rhs_sp), cg_max_iter, tol=tol, atol=atol)
print("AMGCG took", timeit.default_timer()-t0, 's after', len(res_amgcg), 'iterations', f'to {res_amgcg[-1]}')
################
# CG
################
t0 = timeit.default_timer()
x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), cg_max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
print("CG took", timeit.default_timer()-t0, 's after', len(res_cg), 'iterations', f'to {res_cg[-1]}')

###############
# CG GPU
###############
rhs_cp, A_cp = cp.array(rhs_sp, dtype=np.float64), cpsp.csr_matrix(A_sp, dtype=np.float64)
x_cg_cp, res_cg_cp = None, None
def cuda_benchmark_cg():
    global x_cg_cp, res_cg_cp
    x_cg_cp, res_cg_cp = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), cg_max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
result = cuda_benchmark(cuda_benchmark_cg, n_repeat=1)
print("CUDA CG took", result.gpu_times[0][0], 's after', len(res_cg_cp), 'iterations', f'to {res_cg_cp[-1]}')


def dcdm_predict(dcdm_model):
    global flags
    def predict(r):
        with torch.no_grad():
            r = normalize(r, dim=0)
            r = r.view(1, 1, N, N)
            x = dcdm_model(r).flatten()
        return x
    return predict

def fluidnet_predict(fluidnet_model, image):
    def predict(r):
        with torch.no_grad():
            r = normalize(r, dim=0)
            x = fluidnet_model.eval_forward(image.view((1,)+(N,)*DIM), r.view((1, 1)+(N,)*DIM)).flatten()
        return x
    return predict


x_fluidnet_res, res_fluidnet_res = None, None
x_fluidnet_eng, res_fluidnet_eng = None, None
x_fluidnet_scale2, res_fluidnet_scale2 = None, None
x_fluidnet_scaleA, res_fluidnet_scaleA = None, None

def torch_benchmark_dcdm_res(model):
    global x_fluidnet_res, res_fluidnet_res
    x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model, flags), pcg_max_iter, tol=tol, atol=atol)
def torch_benchmark_dcdm_eng(model):
    global x_fluidnet_eng, res_fluidnet_eng
    x_fluidnet_eng, res_fluidnet_eng = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model, flags), pcg_max_iter, tol=tol, atol=atol)
def torch_benchmark_dcdm_scaled2(model):
    global x_fluidnet_scale2, res_fluidnet_scale2
    x_fluidnet_scale2, res_fluidnet_scale2 = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model, flags), pcg_max_iter, tol=tol, atol=atol)
def torch_benchmark_dcdm_scaledA(model):
    global x_fluidnet_scaleA, res_fluidnet_scaleA
    x_fluidnet_scaleA, res_fluidnet_scaleA = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model, flags), pcg_max_iter, tol=tol, atol=atol)
def torch_timer(loss):
    return torch_benchmark.Timer(
    stmt=f'torch_benchmark_dcdm_{loss}(model)',
    setup=f'from __main__ import torch_benchmark_dcdm_{loss}',
    globals={'model':eval(f"fluidnet_model_{loss}")})

timer_res = torch_timer('res')
result_res = timer_res.timeit(1)
print(f"{NN} Small SM D4 took", result_res.times[0], 's after', len(res_fluidnet_res), 'iterations', f'to {res_fluidnet_res[-1]}')

# timer_eng = torch_timer('eng')
# result_eng = timer_eng.timeit(1)
# print(f"{NN} Small SM took", result_eng.times[0], 's after', len(res_fluidnet_eng), 'iterations', f'to {res_fluidnet_eng[-1]}')

# timer_scaled2 = torch_timer('scaled2')
# result_scaled2 = timer_scaled2.timeit(1)
# print(f"{NN} FluidNet took", result_scaled2.times[0], 's after', len(res_fluidnet_scale2), 'iterations', f'to {res_fluidnet_scale2[-1]}')

# timer_scaledA = torch_timer('scaledA')
# result_scaledA = timer_scaledA.timeit(1)
# print(f"{NN} Small SM D4 took", result_scaledA.times[0], 's after', len(res_fluidnet_scaleA), 'iterations', f'to {res_fluidnet_scaleA[-1]}')

import matplotlib.pyplot as plt
plt.plot(res_fluidnet_res, label=f'{NN} small_sm_d4')
plt.plot(res_amgcg, label='amgcg')
# plt.plot(res_fluidnet_eng, label=f'{NN} small_sm')
# plt.plot(res_fluidnet_scale2, label=f'{NN} fluidnet')
# plt.plot(res_fluidnet_scaleA, label=f'{NN} small_sm_d4')
plt.plot(res_cg, label='cg')
plt.plot(res_cg_cp, label='cuda cg')
if norm_type == 'l2': plt.yscale('log')
plt.title(f"{norm_type} norm VS Iterations")
plt.legend()
plt.savefig("test_loss.png")
