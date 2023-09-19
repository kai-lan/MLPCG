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
    def run_frames(self, scene, shape, frames, output=None):
        results = {'amgcl_time': [], 'amgcl_iters': [],
                    'ic_time': [], 'ic_iters': [],
                    'cg_time': [], 'cg_iters': [],
                    'mlpcg_time': [], 'mlpcg_iters': []}
        for i, frame in enumerate(frames):
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


            out = f"{frame:<4}"
            title = f"{'Frames':<4}"

            if self.solvers['AMGCL']:
                x_amgcl, (iters_amgcl, tot_time, res_amgcl) = AMGCL_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_cg_iters, tol=self.rel_tol)
                results['amgcl_time'].append(tot_time)
                results['amgcl_iters'].append(iters_amgcl)
                print("AMGCL took", tot_time, 's after', iters_amgcl, 'iterations', f'to {res_amgcl}')
                out += f", {iters_amgcl:^4}, {tot_time:>6.4f}"
                if i == 0: title += f", {'AMG':^4}, {'':>6}"

            if self.solvers['IC']:
                x_ic, (iters_ic, tot_time, res_ic) = IC_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_ic_iters, tol=self.rel_tol)
                results['ic_time'].append(tot_time)
                results['ic_iters'].append(iters_ic)
                print("IC took", tot_time, 's after', iters_ic, 'iterations', f'to {res_ic}')
                out += f", {iters_ic:^4}, {tot_time:>6.4f}"
                if i == 0: title += f", {'IC':^4}, {'':>6}"

            if self.solvers['CG']:
                rhs_cp, A_cp = cp.array(rhs_comp, dtype=np.float64), cpsp.csr_matrix(A_comp, dtype=np.float64)
                result = cuda_benchmark(self.benchmark_cuda_cg_func(rhs_cp, A_cp, cp.zeros_like(rhs_cp)), n_repeat=1)
                x_cg, iters = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), self.max_cg_iters, tol=self.rel_tol)
                results['cg_time'].append(result.gpu_times[0][0])
                results['cg_iters'].append(iters)
                print("CUDA CG took", result.gpu_times[0][0], 's after', iters, 'iterations')
                out += f", {iters:^4}, {result.gpu_times[0][0]:>6.4f}"
                if i == 0: title += f", {'CG':^4}, {'':>6}"

            if self.solvers['MLPCG']:
                flags_sp = convert_to_binary_images(flags_sp, num_imgs)
                A = torch.sparse_csr_tensor(A_comp.indptr, A_comp.indices, A_comp.data, A_comp.shape, dtype=torch.float64, device=device)
                rhs = torch.tensor(rhs_comp, dtype=torch.float64, device=device)
                flags = torch.tensor(flags_sp, dtype=pcg_dtype, device=device).view(num_imgs, *shape)
                fluid_cells = torch.from_numpy(fluid_cells).to(device)
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
                out += f", {iters:^4}, {total_time:>6.4f}"
                if i == 0: title += f", {'ML':^4}, {'':>6}"
            if output is not None:
                with open(output_file, 'a') as f:
                    if i == 0: f.write(title + '\n')
                    f.write(out + '\n')
        avg = f"{'Avg':<4}"
        if solvers['AMGCL']:
            avg += f", {np.mean(results['amgcl_iters']):^4.4f}, {np.mean(results['amgcl_time']):>6.4f}"
        if solvers['IC']:
            avg += f", {np.mean(results['ic_iters']):^4.4f}, {np.mean(results['ic_time']):>6.4f}"
        if solvers['CG']:
            avg += f", {np.mean(results['cg_iters']):^4.4f}, {np.mean(results['cg_time']):>6.4f}"
        if solvers['MLPCG']:
            avg += f", {np.mean(results['mlpcg_iters']):^4.4f}, {np.mean(results['mlpcg_time']):>6.4f}\n"
        if output is not None:
            with open(output_file, 'a') as f:
                f.write(avg)

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
        "AMGCL": False,
        "IC": False,
        "CG": False
        }

DIM = 3

N = 256
shape = (N,) + (N,)*(DIM-1)
device = torch.device('cuda')
frames = range(1, 201)
# scene = f'standing_pool_scooping_N{N}_200_{DIM}D'
# scene = f'dambreak_pillars_N{N}_N{2*N}_200_{DIM}D'
scene = f'waterflow_ball_N{N}_200_3D'


NN = 128
num_mat = 10
num_ritz = 1600
num_rhs = 800
num_imgs = 3
num_levels = 3
model_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_mixedBCs_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_res_imgs{num_imgs}_lr0.0001_50.tar")
model = SmallSMModelDn3D(num_levels, num_imgs)
model.move_to(device)
state_dict = torch.load(model_file, map_location=device)['model_state_dict']
model.load_state_dict(state_dict)
model.eval()

output_file = model_file.replace("checkpt", f"test_{scene}").replace(".tar", ".txt")
with open(output_file, 'w') as f:
    f.write('')
tests = Tests(model, solvers, 1e-6)
results = tests.run_frames(scene, shape, frames, output=output_file)


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

