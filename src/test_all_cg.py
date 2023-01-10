# This is an example test code for the paper
# The test code solves the linear system A x = b, where A is pressure matrix, and b is the velocity divergence
# Both A and b is creted from a simulation. We provide various simulations, such as smoke plume, rotating fluid, etc,
# for reader to pick to test.
# A and b also depends on the frame number

# Load the required libraries
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np
import torch
from torch.nn.functional import normalize
import scipy.sparse as sparse
import time
import argparse
from train import DCDM
import conjugate_gradient as cg
import read_data as hf


# Command variables
N = 64 # choices=[64, 128, 256, 384]
k = 64 # 64, 128
DIM = 3 # 2, 3
example_name = "smoke_passing_bunny" # choices=["rotating_fluid", "smoke_passing_bunny"]

frame_number = 10

max_cg_iter = 1000

tol = 1.0e-4

verbose_dcdm = False

verbose_icpcg = False

data_path = os.path.join(dir_path, "..", "dataset_mlpcg")

# if matrix does not change in the example, use the matrix for the first frame.
if example_name in ["rotating_fluid", "smoke_passing_bunny"]:
    matrix_frame_number = 1
else:
    matrix_frame_number = frame_number

#Decide which example to run for: SmokePlume, ...
# Smoke
test_smoke_path = os.path.join(data_path, "test_matrices_and_vectors", "N64", "smoke_passing_bunny")
b_smoke = hf.load_vector(os.path.join(test_smoke_path, "div_v_star1.bin"), dtype='d').astype(np.float32)
A_smoke = hf.readA_sparse(64, os.path.join(test_smoke_path, "matrixA_1.bin"), DIM=3)
CG_smoke = cg.ConjugateGradientSparse(A_smoke)

# test_rotate_path = os.path.join(data_path, "test_matrices_and_vectors", "N128", "rotating_fluid")
# b_rotate = hf.load_vector(os.path.join(test_rotate_path, "div_v_star_2.bin"))

###################
# DCDM
###################
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
