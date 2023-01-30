import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(dir_path, "..", "lib"))

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sparse
import time
from train import FluidNet
import conjugate_gradient as cg
import read_data as hf


# Command variables
N = 64
DIM = 2

frame = 800
max_cg_iter = 1000
tol = 1.0e-4

verbose_dcdm = True
verbose_icpcg = False

# Test the dambreak example
test_dambreak_path = os.path.join(dir_path, "..", "data_fluidnet", f"dambreak_{DIM}D_{N}")
model_dambreak_path = os.path.join(dir_path, "..", "data_fluidnet", f"output_{DIM}D_{N}")

b_dambreak = hf.load_vector(os.path.join(test_dambreak_path, f"div_v_star_{frame}.bin"), dtype='d').astype(np.float32)
A_dambreak = hf.readA_sparse(64, os.path.join(test_dambreak_path, f"A_{frame}.bin"), DIM=2).astype(np.float32)
CG_dambreak = cg.ConjugateGradientSparse(A_dambreak)
flags_dambreak = hf.read_flags(os.path.join(test_dambreak_path, f"flags_{frame}.bin"))
flags_dambreak = torch.tensor(flags_dambreak, dtype=torch.float32)

###################
# FluidNet
###################
print("Loading fluidnet model")
model = FluidNet()
model_file = os.path.join(model_dambreak_path, "model_Sun-Jan-29-22:25:46-2023.pth")
model.load_state_dict(torch.load(model_file))
model.eval()

def model_predict(r):
    global model
    with torch.no_grad():
        r = torch.tensor(r, dtype=torch.float32)
        r = F.normalize(r, dim=0)
        r = torch.stack([r, flags_dambreak], dim=1)
        r = r.view(*((1, 2,)+(N,)*DIM))
        x = model(r).flatten().numpy()
    return x

print("--------------")
t0 = time.time()
max_dcdm_iter = 100
x_sol, res_arr = CG_dambreak.dcdm(b_dambreak, np.zeros(b_dambreak.shape), model_predict, max_dcdm_iter, tol, False ,verbose_dcdm)
print("FLuidNet took", time.time() - t0, "s.")
exit()
if 1:
    print("Loading model from disk.")
    model = DCDM(DIM)
    model_file = os.path.join(data_path, f"output_{N}_{DIM}D", "model_Mon-Jan--9-01:52:28-2023.pth")
    model.load_state_dict(torch.load(model_file))
    model.eval()
    print("Loaded trained model from disk")
    def model_predict(r):
        global model
        with torch.no_grad():
            r = torch.tensor(r, dtype=torch.float32)
            r = normalize(r, dim=0)
            r = r.view(*((1, 1,)+(N,)*DIM))
            x = model(r).flatten().numpy()
        return x
    print("DGCM is running...")
    t0 = time.time()
    max_dcdm_iter = 100
    x_sol, res_arr= CG_smoke.dcdm(b_smoke, np.zeros(b_smoke.shape), model_predict, max_dcdm_iter, tol, False ,verbose_dcdm)
    time_cg_ml = time.time() - t0
    print("DGCM took ", time_cg_ml, " secs.")

if 0:
    print("CG is running...")
    t0 = time.time()
    x_sol_cg, res_arr_cg = CG_smoke.cg_normal(np.zeros(b_smoke.shape), b_smoke, max_cg_iter, tol, True)
    time_cg = time.time() - t0
    print("CG took ",time_cg, " secs")

if 0:
    print("DeflatedPCG is running")
    t0=time.time()
    x_sol_cg, res_arr_cg = CG_smoke.deflated_pcg(b_smoke, max_cg_iter, tol, 50, True)
    time_cg = time.time() - t0
    print("Deflated PCG took ",time_cg, " secs")

if 0:
    print("icpcg is running...")
    t0=time.time()
    L, D = CG_smoke.ldlt()
    time_ldlt_creation = time.time() - t0
    print("L and D computed in ", time_ldlt_creation, " seconds.")
    l_name = data_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "_matrix_L_fn"+str(matrix_frame_number)+".npz"
    d_name = data_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "_matrix_D_fn"+str(matrix_frame_number)+".npz"
    sparse.save_npz(l_name, L)
    sparse.save_npz(d_name, D)
    #L = sparse.load_npz(l_name)
    #D = sparse.load_npz(d_name)

    print("icpcg PCG is running...")
    t0 = time.time()
    x,res_arr_cg  = CG_smoke.ldlt_pcg(L, D, b, max_cg_iter, tol, verbose_icpcg)
    time_ldlt_pcg = time.time() - t0
    print("icpcg PCG took ", time_ldlt_pcg, " seconds.")
