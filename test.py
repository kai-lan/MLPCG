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
    def output_fluid_cells(self, scene, shape, frames, output=None):
        scene_path = os.path.join(DATA_PATH, f"{scene}")
        running_bunny = 'smoke_bunny' in scene
        for i, frame in enumerate(frames):
            print("Testing frame", frame, "scene", scene)
            if running_bunny:
                flags_sp = read_flags(os.path.join(scene_path, f"flags_1.bin"))
            else:
                flags_sp = read_flags(os.path.join(scene_path, f"flags_{frame}.bin"))
            fluid_cells = np.argwhere(flags_sp == FLUID).ravel()
            size = len(fluid_cells)
            print("Number of fluid cells", size, size / np.prod(shape))
            if output:
                with open(output, 'a') as f:
                    f.write(f"{frame:>4}, {size:>10}\n")
    def test_cholesky(self, scene, shape, frames, output=None):
        scene_path = os.path.join(DATA_PATH, f"{scene}")
        for i, frame in enumerate(frames):
            print("Testing frame", frame, "scene", scene)
            flags = read_flags(os.path.join(scene_path, f"flags_{frame}.bin"))
            rhs = load_vector(f"{scene_path}/div_v_star_{frame}.bin")
            A = readA_sparse(f"{scene_path}/A_{frame}.bin", sparse_type='csc')
            if len(rhs) == np.prod(shape):
                A = compressedMat(A, flags)
                rhs = compressedVec(rhs, flags)
            A_upper = sparse.triu(A, format='csc')
            # A_lower = sparse.tril(A, format='csc')
            fluid_cells = np.argwhere(flags == FLUID).ravel()
            size = len(fluid_cells)
            try:
                start = time.time()
                x = Cholesky_meshfem(rhs, A_upper)
                # x = Cholesky_cuda(rhs, A_upper)
                # x = Cholesky_scikit_sparse(rhs, A_lower)
                total_time = time.time()-start
                r = rhs - A @ x
                # r = b.cpu().numpy() -  A @ x.cpu().numpy()
                norm = np.linalg.norm(r) / np.linalg.norm(rhs)
                print('Residual', norm)
                print('Total time', total_time)
                if output:
                    with open(output, 'a') as f:
                        f.write(f"{frame:>4}, {size:>10}, {total_time:>6.2f}\n")
            except:
                print("Failed to solve")
                total_time = ' '
                if output:
                    with open(output, 'a') as f:
                        f.write(f"{frame:>4}, {size:>10}, {' ':>6}\n")
    def run_frames(self, scene, shape, frames, output=None):
        results = {'amgcl_time': [], 'amgcl_iters': [],
                    'ic_time': [], 'ic_iters': [],
                    'cg_time': [], 'cg_iters': [],
                    'mlpcg_time': [], 'mlpcg_iters': []}
        scene_path = os.path.join(DATA_PATH, f"{scene}")
        running_bunny = 'smoke_bunny' in scene
        for i, frame in enumerate(frames):
            print("Testing frame", frame, "scene", scene)
            if running_bunny:
                A_sp = readA_sparse(os.path.join(scene_path, f"A_1.bin")).astype(np.float64)
            else:
                A_sp = readA_sparse(os.path.join(scene_path, f"A_{frame}.bin")).astype(np.float64)
            rhs_sp = load_vector(os.path.join(scene_path, f"div_v_star_{frame}.bin")).astype(np.float64)
            if running_bunny:
                flags_sp = read_flags(os.path.join(scene_path, f"flags_1.bin"))
            else:
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
                if i == 0: title += f", {'AMG':>4}, {'':>6}"

            if self.solvers['IC']:
                x_ic, (iters_ic, tot_time, res_ic) = IC_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_ic_iters, tol=self.rel_tol)
                results['ic_time'].append(tot_time)
                results['ic_iters'].append(iters_ic)
                print("IC took", tot_time, 's after', iters_ic, 'iterations', f'to {res_ic}')
                out += f", {iters_ic:^4}, {tot_time:>6.4f}"
                if i == 0: title += f", {'IC':>4}, {'':>6}"

            if self.solvers['CG']:
                rhs_cp, A_cp = cp.array(rhs_comp, dtype=np.float64), cpsp.csr_matrix(A_comp, dtype=np.float64)
                result = cuda_benchmark(self.benchmark_cuda_cg_func(rhs_cp, A_cp, cp.zeros_like(rhs_cp)), n_repeat=3, n_warmup=2)
                x_cg, iters = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), self.max_cg_iters, tol=self.rel_tol)
                r_cg = cp.linalg.norm(rhs_cp - A_cp @ x_cg) / cp.linalg.norm(rhs_cp)
                results['cg_time'].append(result.gpu_times[0][0])
                results['cg_iters'].append(iters)
                print("CUDA CG took", result.gpu_times[0][0], 's after', iters, 'iterations', f"to {r_cg.item()}")
                out += f", {iters:^4}, {result.gpu_times[0][0]:>6.4f}"
                if i == 0: title += f", {'CG':>4}, {'':>6}"

            if self.solvers['MLPCG']:
                flags_sp = convert_to_binary_images(flags_sp, num_imgs)
                A = torch.sparse_csr_tensor(A_comp.indptr, A_comp.indices, A_comp.data, A_comp.shape, dtype=torch.float64, device=device)
                rhs = torch.tensor(rhs_comp, dtype=torch.float64, device=device)
                flags = torch.tensor(flags_sp, dtype=pcg_dtype, device=device).view(num_imgs, *shape)
                fluid_cells = torch.from_numpy(fluid_cells).to(device)
                predict = self.model_predict(model, flags, fluid_cells)

                for _ in range(2): # warm up
                    npsd(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)

                total_time = 0.0
                steps = 3
                for _ in range(steps):
                    start_time = time.perf_counter()
                    npsd(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)

                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                total_time /= steps
                x_mlpcg, iters, timer, res = npsd(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)

                results['mlpcg_time'].append(total_time)
                results['mlpcg_iters'].append(iters)
                print(f"MLPCG took", total_time, 's after', iters, f"iterations to {res}")
                timer.report()
                out += f", {iters:^4}, {total_time:>6.4f}"
                if i == 0: title += f", {'ML':>4}, {'':>6}"
            if output is not None:
                with open(output, 'a') as f:
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
            with open(output, 'a') as f:
                f.write(avg)

        return results

    def profile_scene(self, scene, shape, frames, output=None, output1=None):
        res_profile = []
        time_profile = []
        def callback(res, time):
            nonlocal res_profile, time_profile
            res_profile.append(res)
            time_profile.append(time)

        scene_path = os.path.join(DATA_PATH, f"{scene}")
        running_bunny = 'smoke_bunny' in scene
        for i, frame in enumerate(frames):
            print("Testing frame", frame, "scene", scene)
            if running_bunny:
                A_sp = readA_sparse(os.path.join(scene_path, f"A_1.bin")).astype(np.float64)
            else:
                A_sp = readA_sparse(os.path.join(scene_path, f"A_{frame}.bin")).astype(np.float64)
            rhs_sp = load_vector(os.path.join(scene_path, f"div_v_star_{frame}.bin")).astype(np.float64)
            if running_bunny:
                flags_sp = read_flags(os.path.join(scene_path, f"flags_1.bin"))
            else:
                flags_sp = read_flags(os.path.join(scene_path, f"flags_{frame}.bin"))
            fluid_cells = np.argwhere(flags_sp == FLUID).ravel()
            # compressed A and rhs
            if len(rhs_sp) == np.prod(shape):
                A_comp = compressedMat(A_sp, flags_sp)
                rhs_comp = compressedVec(rhs_sp, flags_sp)
            else:
                A_comp = A_sp
                rhs_comp = rhs_sp

            if self.solvers['AMGCL']:
                x_amgcl, (iters_amgcl, tot_time, res_amgcl) = AMGCL_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_cg_iters, tol=self.rel_tol, verbose=True)
                if output1:
                    with open(output1, 'a') as f:
                        f.write(f"{frame}, {iters_amgcl:>4}, {tot_time:>6.4f}\n")

                print("AMGCL took", tot_time, 's after', iters_amgcl, 'iterations', f'to {res_amgcl}')


            elif self.solvers['IC']:
                x_ic, (iters_ic, tot_time, res_ic) = IC_CUDA(rhs_comp, A_comp, np.zeros_like(rhs_comp), self.max_ic_iters, tol=self.rel_tol, verbose=True)
                if output1:
                    with open(output1, 'a') as f:
                        f.write(f"{frame}, {iters_ic:>4}, {tot_time:>6.4f}\n")
                print("IC took", tot_time, 's after', iters_ic, 'iterations', f'to {res_ic}')

            elif self.solvers['CG']:
                rhs_cp, A_cp = cp.array(rhs_comp, dtype=np.float64), cpsp.csr_matrix(A_comp, dtype=np.float64)
                result = cuda_benchmark(self.benchmark_cuda_cg_func(rhs_cp, A_cp, cp.zeros_like(rhs_cp)), n_repeat=1, n_warmup=2)
                x_cg, iters = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), self.max_cg_iters, tol=self.rel_tol, callback=callback)

                if output1:
                    with open(output1, 'a') as f:
                        f.write(f"{frame}, {iters:>4}, {result.gpu_times[0][0]:>6.4f}\n")

                print("CUDA CG took", result.gpu_times[0][0], 's after', iters, 'iterations')

            elif self.solvers['MLPCG']:
                flags_sp = convert_to_binary_images(flags_sp, num_imgs)
                A = torch.sparse_csr_tensor(A_comp.indptr, A_comp.indices, A_comp.data, A_comp.shape, dtype=torch.float64, device=device)
                rhs = torch.tensor(rhs_comp, dtype=torch.float64, device=device)
                flags = torch.tensor(flags_sp, dtype=pcg_dtype, device=device).view(num_imgs, *shape)
                fluid_cells = torch.from_numpy(fluid_cells).to(device)
                predict = self.model_predict(model, flags, fluid_cells)

                for _ in range(2): # warm up
                    npsd(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol)
                x_mlpcg, iters, timer, res = npsd(rhs, A, torch.zeros_like(rhs), predict, self.max_mlpcg_iters, tol=self.rel_tol, callback=callback)
                print(f"MLPCG took", timer.top_level_clocks['Total'].tot_time, 's after', iters, f"iterations to {res}")

            for i in reversed(range(len(time_profile))):
                time_profile[i] -= time_profile[0]
            line1 = f"{frame:<4}"
            line2 = f"{frame:<4}"
            for res, time in zip(res_profile, time_profile):
                line1 += f", {res:>8.4}"
                line2 += f", {time:>8.4}"
            line1 += "\n"
            line2 += "\n"

            if output:
                with open(output, 'a') as f:
                    f.write(line1)
                    f.write(line2)
            res_profile, time_profile = [], []

if len(sys.argv) > 1:
    solver = sys.argv[1]

solvers = {
        "MLPCG": True,
        "AMGCL": False,
        "IC": False,
        "CG": False
        }

DIM = 3

N = 128
N2 = 256
device = torch.device('cuda')
frames = range(200, 201)
# frames = [200]
# scene = f'dambreak_bunny_N{N//2}_N{N}_200_{DIM}D'
# scene = f'dambreak_pillars_N{N}_N{2*N}_200_{DIM}D'
scene = f'waterflow_ball_N{N2}_200_3D'
# scene = f'standing_scooping_N{N}_200_3D'

shape = (N2,) * DIM

NN = 128
num_mat = 10
num_ritz = 1600
num_rhs = 800
num_imgs = 3
num_levels = 4
model_file = os.path.join(OUT_PATH, f"output_{DIM}D_{NN}", f"checkpt_mixedBCs_M{num_mat}_ritz{num_ritz}_rhs{num_rhs}_imgs{num_imgs}_lr0.0001_30.tar")
model = SmallSMModelDn3D(num_levels, num_imgs)
model.move_to(device)
state_dict = torch.load(model_file, map_location=device)['model_state_dict']

for i in range(num_levels):
    state_dict[f'c0.{i}.bias'] = state_dict[f'c0.{i}.bias'].mean(dim=0, keepdim=True)
    state_dict[f'c1.{i}.bias'] = state_dict[f'c1.{i}.bias'].mean(dim=0, keepdim=True)
    state_dict[f'c0.{i}.weight'] = state_dict[f'c0.{i}.weight'].mean(dim=0, keepdim=True)
    state_dict[f'c1.{i}.weight'] = state_dict[f'c1.{i}.weight'].mean(dim=0, keepdim=True)

model.load_state_dict(state_dict)
model.eval()

output_file1 = model_file.replace("checkpt", f"test_{scene}").replace(".tar", ".txt")
# output_file = model_file.replace("checkpt", f"res_iters_{bc}").replace(".tar", ".txt")
# with open(output_file, 'w') as f:
    # f.write('')
with open(output_file1, 'w') as f:
    f.write('')
tests = Tests(model, solvers, 1e-6)
results = tests.run_frames(scene, shape, frames, output=output_file1)


    ################
    # Summary
    ################
    # print('\nOn average\n')
    # if solvers['AMGCL']:
    #     print('AMGCL took', np.mean(results['amgcl_iters']), 'iters', np.mean(results['amgcl_time']), 's')
    # if solvers['IC']:
    #     print('IC took', np.mean(results['ic_iters']), 'iters', np.mean(results['ic_time']), 's')
    # if solvers['CG']:
    #     print('CG took', np.mean(results['cg_iters']), 'iters', np.mean(results['cg_time']), 's')
    # if solvers['MLPCG']:
    #     print('MLPCG took', np.mean(results['mlpcg_iters']), 'iters', np.mean(results['mlpcg_time']), 's')

