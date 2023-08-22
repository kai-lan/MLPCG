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

pcg_precision = torch.float32
torch.set_default_dtype(pcg_precision)

DIM = 3
norm_type = 'l2'
num_imgs = 3

N = 128
shape = (N,) + (N,)*(DIM-1)
# train_matrices = set(np.load(f"{OUT_PATH}/output_{DIM}D_{N}/matrices_trained_50.npy"))
# frame = list(frames)[80] # Random frame
# frames = range(1, 201)
# frames = np.linspace(1, 200, 20, dtype=int)
frames = [100]

device = torch.device('cuda')
amgcg_time, amgcg_iters = [], []
amgcl_time, amgcl_iters = [], []
cg_time, cg_iters = [], []
mlpcg_time, mlpcg_iters = [], []

if DIM == 2:
    scene = f'standing_scooping_N{N}_200'
else:
    scene = f'waterflow_rotating_cube_N{N}_200_{DIM}D'

NN = 128
num_mat = 10
num_ritz = 1600
num_rhs = 800
tests = {
    "MLPCG": True,
    "AMGCG": False,
    "AMGCL": True,
    "CG": True,
}
# fluidnet_model_res_file = os.path.join(OUT_PATH, f"output_single_{DIM}D_{NN}", "checkpt_dambreak_frame_1_rhs_1600_2D_linear.tar")

fluidnet_model_res_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_mixedBCs_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_imgs{num_imgs}_lr0.0001.tar")

fluidnet_model_res = SmallSMModelDn3D(3, num_imgs) if DIM == 3 else SmallSMModelDn(6, num_imgs)
fluidnet_model_res.move_to(device)
state_dict = torch.load(fluidnet_model_res_file, map_location=device)['model_state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('.KL', '')] = state_dict.pop(key)
fluidnet_model_res.load_state_dict(state_dict)
fluidnet_model_res.eval()

verbose = False
cg_max_iter = 1500
pcg_max_iter = 500
tol = 1e-4
atol = 1e-10 # safe guard for small rhs

def fluidnet_predict(fluidnet_model, image):
    def predict(r):
        with torch.no_grad():
            r = normalize(r, dim=0)
            x = fluidnet_model.eval_forward(image.view((num_imgs,)+shape).float(), r.view((1, 1)+shape).float()).flatten().double()
        return x
    return predict

x_fluidnet_res, res_fluidnet_res = None, None
def torch_benchmark_dcdm_res(model):
    global x_fluidnet_res, res_fluidnet_res
    x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), fluidnet_predict(model, flags), pcg_max_iter, tol=tol, atol=atol)
def torch_timer(loss):
    return torch_benchmark.Timer(
    stmt=f'torch_benchmark_dcdm_{loss}(model)',
    setup=f'from __main__ import torch_benchmark_dcdm_{loss}',
    globals={'model':eval(f"fluidnet_model_{loss}")})

x_cg, res_cg = None, None
def cuda_benchmark_cg():
    global x_cg, res_cg
    x_cg, res_cg = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), cg_max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)

for frame in frames:
    print("Testing frame", frame, "scene", scene)
    dambreak_path = os.path.join(DATA_PATH, f"{scene}") #_smoke include boundary
    A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin")).astype(np.float64)
    rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin")).astype(np.float64)
    flags_sp = read_flags(os.path.join(dambreak_path, f"flags_{frame}.bin"))

    # A_sp = readA_sparse("cxx_src/test_data/A_999.bin")
    # rhs_sp = load_vector("cxx_src/test_data/b_999.bin")
    # flags_sp = read_flags("cxx_src/test_data/flags_999.bin")

    # compressed A and rhs
    A_comp = compressedMat(A_sp, flags_sp)
    rhs_comp = compressedVec(rhs_sp, flags_sp)

    flags_sp = convert_to_binary_images(flags_sp, num_imgs)

    A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float64, device=device)
    rhs = torch.tensor(rhs_sp, dtype=torch.float64, device=device)
    flags = torch.tensor(flags_sp, dtype=torch.float64, device=device)

    ###############
    # AMGCG (CPU)
    ###############
    if tests['AMGCG']:
        t0 = timeit.default_timer()
        res_amgcg = 0
        for _ in range(10):
            x_amgcg, res_amgcg = AMGCG(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol)
        t_amgcg = (timeit.default_timer()-t0) / 10
        amgcg_time.append(t_amgcg)
        amgcg_iters.append(len(res_amgcg))
        print("AMGCG took", t_amgcg, 's after', len(res_amgcg), 'iterations', f'to {res_amgcg[-1]}')

    ###############
    # AMGCGL (CPU)
    ###############
    if tests['AMGCL']:
        if rhs.is_cuda:
            t0 = timeit.default_timer()
            res_amgcl = 0
            for _ in range(50):
                x_amgcl, (iters_amgcl, res_amgcl) = AMGCL_VEXCL(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol)
            t_amgcl = (timeit.default_timer()-t0) / 50
            amgcl_time.append(t_amgcl)
            amgcl_iters.append(iters_amgcl)
            print("AMGCL took", t_amgcl, 's after', iters_amgcl, 'iterations', f'to {res_amgcl}')
        else:
            t0 = timeit.default_timer()
            res_amgcl = 0
            for _ in range(50):
                x_amgcl, (iters_amgcl, res_amgcl) = AMGCL(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol)
            t_amgcl = (timeit.default_timer()-t0) / 50
            amgcl_time.append(t_amgcl)
            amgcl_iters.append(iters_amgcl)
            print("AMGCL took", t_amgcl, 's after', iters_amgcl, 'iterations', f'to {res_amgcl}')

    ################
    # CG or CUDA CG
    ################
    if tests['CG']:
        if rhs.is_cuda:
            rhs_cp, A_cp = cp.array(rhs_sp, dtype=np.float64), cpsp.csr_matrix(A_sp, dtype=np.float64)
            # rhs_cp, A_cp = cp.array(rhs_comp, dtype=np.float64), cpsp.csr_matrix(A_comp, dtype=np.float64)

            result = cuda_benchmark(cuda_benchmark_cg, n_repeat=1)
            cg_time.append(result.gpu_times[0][0])
            cg_iters.append(len(res_cg))
            print("CUDA CG took", result.gpu_times[0][0], 's after', len(res_cg), 'iterations', f'to {res_cg[-1]}')
        else:
            t0 = timeit.default_timer()
            # x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), cg_max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
            x_cg, res_cg = CG(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
            t_cg = timeit.default_timer()-t0
            cg_time.append(t_cg)
            cg_iters.append(len(res_cg))
            print("CG took", t_cg, 's after', len(res_cg), 'iterations', f'to {res_cg[-1]}')

    if tests['MLPCG']:
        timer_res = torch_timer('res')
        result_res = timer_res.timeit(1)
        mlpcg_time.append(result_res.times[0])
        mlpcg_iters.append(len(res_fluidnet_res))
        print(f"MLPCG", result_res.times[0], 's after', len(res_fluidnet_res), 'iterations', f'to {res_fluidnet_res[-1]}')

    print()

################
# Summary
################
print('\nOn average\n')
# print('AMG took', np.mean(amgcg_iters), 'iters', np.mean(amgcg_time), 's')
print('AMGCL took', np.mean(amgcl_iters), 'iters', np.mean(amgcl_time), 's')
print('CG took', np.mean(cg_iters), 'iters', np.mean(cg_time), 's')
print('MLPCG took', np.mean(mlpcg_iters), 'iters', np.mean(mlpcg_time), 's')

output_file = fluidnet_model_res_file.replace("checkpt", f"test_{scene}").replace(".tar", ".txt")
with open(output_file, 'w') as f:
    for i in range(len(mlpcg_iters)):
        f.write(f"{frames[i]:<4}, {amgcl_iters[i]:^4}, {amgcl_time[i]:>6.4f}, {cg_iters[i]:^4}, {cg_time[i]:>6.4f}, {mlpcg_iters[i]:^4}, {mlpcg_time[i]:>6.4f}\n")
# import matplotlib.pyplot as plt
# plt.plot(res_fluidnet_res, label=f'{NN} MLPCG')
# plt.plot(res_amgcg, label='amgcg')
# plt.plot(res_cg, label='cg')
# # plt.plot(res_cg, label='cuda cg')
# if norm_type == 'l2': plt.yscale('log')
# plt.title(f"{norm_type} norm VS Iterations")
# plt.legend()
# plt.savefig("test_loss.png")
