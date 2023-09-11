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
pcg_dtype = torch.float32
torch.set_default_dtype(pcg_dtype)

class Tests:
    def __init__(self,
        model,
        enabled_solvers={
        "MLPCG": True,
        "AMGCL": True,
        "IC": True,
        "CG": True
        },
        rel_tol=1e-6
    ):
        self.device = torch.device('cuda')
        self.model = model
        self.solvers = enabled_solvers
        self.max_cg_iters = 2000
        self.max_amg_iters = 100
        self.max_ic_iters = 500
        self.max_mlpcg_iters = 200
        self.rel_tol = rel_tol

    def model_predict(self, model, image, fluid_cells):
        shape = image.shape[1:]
        def predict(r, timer):
            with torch.no_grad():
                r = normalize(r.to(pcg_dtype), dim=0)
                b = torch.zeros(np.prod(shape), device=device, dtype=pcg_dtype)
                b[fluid_cells] = r
                x = model.eval_forward(image, b.view((1, 1)+shape), timer).flatten().double()
            return x[fluid_cells]
        return predict
    def benchmark_cuda_cg_func(self, rhs_cp, A_cp, x0):
        def cuda_benchmark_cg():
            x_cg, cg_iter = CG_GPU(rhs_cp, A_cp, x0, self.max_cg_iters, tol=self.rel_tol)
        return cuda_benchmark_cg
    def run_frames(self, scene, shape, frames):
        results = {'amgcl_time': [], 'amgcl_iters': [],
                    'ic_time': [], 'ic_iters': [],
                    'cg_time': [], 'cg_iters': [],
                    'mlpcg_time': [], 'mlpcg_iters': []}
        for frame in frames:
            print("Testing frame", frame, "scene", scene)
            scene_path = os.path.join(DATA_PATH, f"{scene}")
            A_sp = readA_sparse(os.path.join(scene_path, f"A_{frame}.bin")).astype(np.float64)
            rhs_sp = load_vector(os.path.join(scene_path, f"div_v_star_{frame}.bin")).astype(np.float64)
            flags_sp = read_flags(os.path.join(scene_path, f"flags_{frame}.bin"))
            fluid_cells = np.argwhere(flags_sp == FLUID).ravel()

            # compressed A and rhs
            if len(rhs_sp) == np.prod(shape):
                A_comp = compressedMat(A_sp, flags_sp)
                rhs_comp = compressedVec(rhs_sp, flags_sp)
            else:
                A_comp = A_sp
                rhs_comp = rhs_sp
            flags_sp = convert_to_binary_images(flags_sp, num_imgs)

            A = torch.sparse_csr_tensor(A_comp.indptr, A_comp.indices, A_comp.data, A_comp.shape, dtype=torch.float64, device=device)
            rhs = torch.tensor(rhs_comp, dtype=torch.float64, device=device)
            flags = torch.tensor(flags_sp, dtype=pcg_dtype, device=device).view(num_imgs, *shape)
            fluid_cells = torch.from_numpy(fluid_cells).to(device)

            if self.solvers['AMGCL']:
                x_amgcl, (iters_amgcl, tot_time, res_amgcl) = AMGCL_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_cg_iters, tol=self.rel_tol)
                results['amgcl_time'].append(tot_time)
                results['amgcl_iters'].append(iters_amgcl)
                print("AMGCL took", tot_time, 's after', iters_amgcl, 'iterations', f'to {res_amgcl}')

            if self.solvers['IC']:
                x_ic, (iters_ic, tot_time, res_ic) = IC_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_ic_iters, tol=self.rel_tol)
                results['ic_time'].append(tot_time)
                results['ic_iters'].append(iters_ic)
                print("IC took", tot_time, 's after', iters_ic, 'iterations', f'to {res_ic}')

            if self.solvers['CG']:
                rhs_cp, A_cp = cp.array(rhs_comp, dtype=np.float64), cpsp.csr_matrix(A_comp, dtype=np.float64)
                result = cuda_benchmark(self.benchmark_cuda_cg_func(rhs_cp, A_cp, cp.zeros_like(rhs_cp)), n_repeat=1)
                x_cg, iters = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), self.max_cg_iters, tol=self.rel_tol)
                results['cg_time'].append(result.gpu_times[0][0])
                results['cg_iters'].append(iters)
                print("CUDA CG took", result.gpu_times[0][0], 's after', iters, 'iterations')

            if self.solvers['MLPCG']:
                predict = self.model_predict(model, flags, fluid_cells)
                for _ in range(10): # warm up
                    dcdm(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)
                total_time = 0.0
                steps = 10
                for _ in range(steps):
                    start_time = time.perf_counter()
                    dcdm(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                total_time /= steps
                x_mlpcg, iters, timer = dcdm(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)
                results['mlpcg_time'].append(total_time)
                results['mlpcg_iters'].append(iters)
                print(f"MLPCG took", total_time, 's after', iters, 'iterations') #, f"to {res_profile['MLPCG'][-1]}")
                timer.report()
        return results
    def run_scenes(self):
        pass
    def profile_one_frame(self):
        res_profile = {"AMGCL":[], "CG":[], "MLPCG":[], "NewMLPCG":[]}
        time_profile = {"AMGCL":[], "CG":[], "MLPCG":[],  "NewMLPCG":[]}
        def callback(res, time, key):
            global res_profile, time_profile
            res_profile[key].append(res)
            time_profile[key].append(time)


solvers = {
        "MLPCG": True,
        "AMGCL": True,
        "IC": True,
        "CG": True
        }

DIM = 3

N = 128
shape = (N,) + (N,)*(DIM-1)
device = torch.device('cuda')
# frames = reversed(np.linspace(1, 200, 5, dtype=int))
frames = [200]

scene = f'waterflow_spiky_torus_N{N}_200_{DIM}D'


NN = 256
num_mat = 97
num_ritz = 1600
num_rhs = 800
num_imgs = 3
num_levels = 3
model_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_mixedBCs_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_imgs{num_imgs}_lr0.0001_from128.tar")
model = SmallSMModelDn3D(num_levels, num_imgs)
model.move_to(device)
state_dict = torch.load(model_file, map_location=device)['model_state_dict']
model.load_state_dict(state_dict)
model.eval()

tests = Tests(model, solvers, 1e-6)
results = tests.run_frames(scene, shape, frames)


################
# Summary
################
print('\nOn average\n')
if solvers['AMGCL']:
    print('AMGCL took', np.mean(results['amgcl_iters']), 'iters', np.mean(results['amgcl_time']), 's')
if solvers['IC']:
    print('IC took', np.mean(results['ic_iters']), 'iters', np.mean(results['ic_time']), 's')
if solvers['CG']:
    print('CG took', np.mean(results['cg_iters']), 'iters', np.mean(results['cg_time']), 's')
if solvers['MLPCG']:
    print('MLPCG took', np.mean(results['mlpcg_iters']), 'iters', np.mean(results['mlpcg_time']), 's')

# output_file = model_file.replace("checkpt", f"test_{scene}").replace(".tar", ".txt")
# with open(output_file, 'w') as f:
#     for i in range(len(mlpcg_iters)):
#         f.write(f"{frames[i]:<4}, {amgcl_iters[i]:^4}, {amgcl_time[i]:>6.4f}, {cg_iters[i]:^4}, {cg_time[i]:>6.4f}, {mlpcg_iters[i]:^4}, {mlpcg_time[i]:>6.4f}\n")
#     f.write(f"{'Avg':<4}, {np.mean(amgcl_iters):^4}, {np.mean(amgcl_time):>6.4f}, {np.mean(cg_iters):^4}, {np.mean(cg_time):>6.4f}, {np.mean(mlpcg_iters):^4}, {np.mean(mlpcg_time):>6.4f}\n")

# import matplotlib.pyplot as plt
# plt.plot(time_profile['MLPCG'], res_profile['MLPCG'], label=f'{NN} MLPCG')
# plt.plot(time_profile['CG'], res_profile['CG'], label='cg')
# plt.yscale('log')
# plt.title(f"{norm_type} norm VS Iterations")
# plt.legend()
# plt.savefig("test_loss.png")
