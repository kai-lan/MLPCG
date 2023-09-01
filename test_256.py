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
from lib.global_clock import *
from torch.nn.functional import normalize
import time, timeit
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_grad_enabled(False) # disable autograd globally
pcg_dtype = torch.float32
torch.set_default_dtype(pcg_dtype)

DIM = 3
norm_type = 'l2'
num_imgs = 3

N = 256
shape = (N,) + (N,)*(DIM-1)

frames = [188]
# frames = np.linspace(1, 200, 10, dtype=int)
# more_frames = np.linspace(12, 188, 9, dtype=int)
# frames = np.concatenate([frames, more_frames])
# frames = range(1, 201)

device = torch.device('cuda')

# res_profile = {"AMGCL":[], "CG":[], "MLPCG":[], "NewMLPCG":[]}
# time_profile = {"AMGCL":[], "CG":[], "MLPCG":[],  "NewMLPCG":[]}

amgcl_time, amgcl_iters = [], []
cg_time, cg_iters = [], []
mlpcg_time, mlpcg_iters = [], []

if DIM == 2:
    scene = f'standing_scooping_N{N}_200'
else:
    scene = f'waterflow_pool_N{N}_200_{DIM}D'

NN = 128
num_mat = 10
num_ritz = 1600
num_rhs = 800
tests = {
    "MLPCG": True,
    "NewMLPCG": False,
    "AMGCG": False,
    "AMGCL": True,
    "CG": True,
}

model_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_mixedBCs_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_imgs{num_imgs}_lr0.0001.tar")

if device == torch.device('cuda'):
    model = SmallSMModelDn3D(3, num_imgs) if DIM == 3 else SmallSMModelDn(6, num_imgs)
else:
    model = SmallSMModelDn3D(3, num_imgs) if DIM == 3 else SmallSMModelDnPY(6, num_imgs)
model.move_to(device)
state_dict = torch.load(model_file, map_location=device)['model_state_dict']
if device == torch.device('cpu'):
    for key in list(state_dict.keys()):
        state_dict[key.replace('.weight', '.KL.weight').replace('.bias', '.KL.bias')] = state_dict.pop(key)
model.load_state_dict(state_dict)
model.eval()

verbose = False
cg_max_iter = 1500
pcg_max_iter = 1500
tol = 1e-6
atol = 1e-10 # safe guard for small rhs

def callback(res, time, key):
    global res_profile, time_profile
    res_profile[key].append(res)
    time_profile[key].append(time)


def model_predict(model, image, fluid_cells):
    shape = image.shape[1:]
    def predict(r, timer):
        with torch.no_grad():
            r = normalize(r.to(pcg_dtype), dim=0)
            b = torch.zeros(np.prod(shape), device=device, dtype=pcg_dtype)
            b[fluid_cells] = r
            x = model.eval_forward(image, b.view((1, 1)+shape), timer).flatten().double()
        return x[fluid_cells]
    return predict

mlpcg_iter = 0
def torch_benchmark_dcdm(model, new=False):
    global mlpcg_iter
    if new:
        x_mlpcg, mlpcg_iter, tot_time = dcdm_new(rhs, A, torch.zeros_like(rhs), model_predict(model, flags, fluid_cells), pcg_max_iter, tol=tol, atol=atol)
    else:
        x_mlpcg, mlpcg_iter, tot_time = dcdm(rhs, A, torch.zeros_like(rhs), model_predict(model, flags, fluid_cells), pcg_max_iter, tol=tol, atol=atol)
def torch_timer(new=False):
    return torch_benchmark.Timer(
    stmt=f'torch_benchmark_dcdm(model, {new})',
    setup=f'from __main__ import torch_benchmark_dcdm',
    globals={'model':eval(f"model")})


cg_iter = 0
def cuda_benchmark_cg():
    global cg_iter
    x_cg, cg_iter = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), cg_max_iter, tol=tol, atol=atol, verbose=verbose)

for frame in frames:
    print("Testing frame", frame, "scene", scene)
    dambreak_path = os.path.join(DATA_PATH, f"{scene}")
    A_sp = readA_sparse(os.path.join(dambreak_path, f"A_{frame}.bin")).astype(np.float64)
    rhs_sp = load_vector(os.path.join(dambreak_path, f"div_v_star_{frame}.bin")).astype(np.float64)
    flags_sp = read_flags(os.path.join(dambreak_path, f"flags_{frame}.bin"))
    fluid_cells = np.argwhere(flags_sp == FLUID).ravel()

    # compressed A and rhs
    # A_comp = compressedMat(A_sp, flags_sp)
    A_comp = A_sp
    # rhs_comp = compressedVec(rhs_sp, flags_sp)
    rhs_comp = rhs_sp
    flags_sp = convert_to_binary_images(flags_sp, num_imgs)

    A = torch.sparse_csr_tensor(A_comp.indptr, A_comp.indices, A_comp.data, A_comp.shape, dtype=torch.float64, device=device)
    rhs = torch.tensor(rhs_comp, dtype=torch.float64, device=device)
    flags = torch.tensor(flags_sp, dtype=pcg_dtype, device=device).view(num_imgs, *shape)
    fluid_cells = torch.from_numpy(fluid_cells).to(device)

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
            x_amgcl, (iters_amgcl, tot_time, res_amgcl) = AMGCL_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol)
            amgcl_time.append(tot_time)
            amgcl_iters.append(iters_amgcl)
            print("AMGCL took", tot_time, 's after', iters_amgcl, 'iterations') #, f'to {res_amgcl}')
        else:
            t0 = timeit.default_timer()
            res_amgcl = 0
            x_amgcl, (iters_amgcl, tot_time, res_amgcl) = AMGCL(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol)
            amgcl_time.append(tot_time)
            amgcl_iters.append(iters_amgcl)
            print("AMGCL took", tot_time, 's after', iters_amgcl, 'iterations') #, f'to {res_amgcl}')

    ################
    # CG or CUDA CG
    ################
    if tests['CG']:
        if rhs.is_cuda:
            # rhs_cp, A_cp = cp.array(rhs_sp, dtype=np.float64), cpsp.csr_matrix(A_sp, dtype=np.float64)
            rhs_cp, A_cp = cp.array(rhs_comp, dtype=np.float64), cpsp.csr_matrix(A_comp, dtype=np.float64)
            # def callback(r, time):
            #     res_profile['CG'].append(r)
            #     time_profile['CG'].append(time)
            result = cuda_benchmark(cuda_benchmark_cg, n_repeat=1)
            # x_cg, iters = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), cg_max_iter, tol=tol, atol=atol, verbose=verbose, callback=callback)

            cg_time.append(result.gpu_times[0][0])
            cg_iters.append(cg_iter)
            print("CUDA CG took", result.gpu_times[0][0], 's after', cg_iter, 'iterations') #, f"to {res_profile['CG'][-1]}")
        else:
            t0 = timeit.default_timer()
            # x_cg, res_cg = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), cg_max_iter, tol=tol, atol=atol, norm_type=norm_type, verbose=verbose)
            x_cg, cg_iter = CG(rhs_comp, A_comp, np.zeros_like(rhs_comp), cg_max_iter, tol=tol, atol=atol, verbose=verbose)
            t_cg = timeit.default_timer()-t0
            cg_time.append(t_cg)
            cg_iters.append(cg_iter)
            print("CG took", t_cg, 's after', cg_iter, 'iterations') #, f'to {res_cg[-1]}')

    if tests['MLPCG']:
        if rhs.is_cuda:
            timer_res = torch_timer()
            result_res = timer_res.timeit(1)
            # def mlpcg_callback(r, time):
            #     res_profile['MLPCG'].append(r)
            #     time_profile['MLPCG'].append(time)
            mlpcg_time.append(result_res.times[0])
            mlpcg_iters.append(mlpcg_iter)
            print(f"MLPCG took", result_res.times[0], 's after', mlpcg_iter, 'iterations') #, f"to {res_profile['MLPCG'][-1]}")
            x_mlpcg, iters, timer = dcdm(rhs, A, torch.zeros_like(rhs), model_predict(model, flags, fluid_cells), pcg_max_iter, tol=tol, atol=atol, callback=None)
            timer.report()
        else:
            t0 = timeit.default_timer()
            x_mlpcg, mlpcg_iter, tot_time = dcdm(rhs, A, torch.zeros_like(rhs), model_predict(model, flags, fluid_cells), pcg_max_iter, tol=tol, atol=atol)
            t_mlpcg = timeit.default_timer()-t0
            mlpcg_time.append(t_mlpcg)
            mlpcg_iters.append(mlpcg_iter)
            print("MLPCG took", t_mlpcg, 's after', mlpcg_iter, 'iterations') #, f'to {res_cg[-1]}')

    # if tests['NewMLPCG']:
    #     timer_res = torch_timer(True)
    #     result_res = timer_res.timeit(1)
    #     def mlpcg_callback(r, time):
    #         res_profile['MLPCG'].append(r)
    #         time_profile['MLPCG'].append(time)
    #     x_mlpcg, iters, tot_time = dcdm_cg(rhs, A, torch.zeros_like(rhs), model_predict(model, flags, fluid_cells), pcg_max_iter, tol=tol, atol=atol, callback=mlpcg_callback)
    #     mlpcg_time.append(result_res.times[0])
    #     mlpcg_iters.append(iters)
    #     print(f"MLPCG", result_res.times[0], 's after', iters, 'iterations', f"to {res_profile['MLPCG'][-1]}")
    #     time_sorted = sorted(tot_time.items(), key=lambda x:x[1], reverse=True)
    #     for item in time_sorted:
    #         print(item)

    print()

################
# Summary
################
print('\nOn average\n')
print('AMGCL took', np.mean(amgcl_iters), 'iters', np.mean(amgcl_time), 's')
print('CG took', np.mean(cg_iters), 'iters', np.mean(cg_time), 's')
print('MLPCG took', np.mean(mlpcg_iters), 'iters', np.mean(mlpcg_time), 's')

output_file = model_file.replace("checkpt", f"test_{scene}").replace(".tar", ".txt")
with open(output_file, 'w') as f:
    for i in range(len(mlpcg_iters)):
        f.write(f"{frames[i]:<4}, {amgcl_iters[i]:^4}, {amgcl_time[i]:>6.4f}, {cg_iters[i]:^4}, {cg_time[i]:>6.4f}, {mlpcg_iters[i]:^4}, {mlpcg_time[i]:>6.4f}\n")
    f.write(f"{'Avg':<4}, {np.mean(amgcl_iters):^4}, {np.mean(amgcl_time):>6.4f}, {np.mean(cg_iters):^4}, {np.mean(cg_time):>6.4f}, {np.mean(mlpcg_iters):^4}, {np.mean(mlpcg_time):>6.4f}\n")

# import matplotlib.pyplot as plt
# plt.plot(time_profile['MLPCG'], res_profile['MLPCG'], label=f'{NN} MLPCG')
# plt.plot(time_profile['CG'], res_profile['CG'], label='cg')
# plt.yscale('log')
# plt.title(f"{norm_type} norm VS Iterations")
# plt.legend()
# plt.savefig("test_loss.png")
