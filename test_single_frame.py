#python3 train.py -N 64 --batch_size 1 --gpu_usage 40 --gpu_choice 0 --dataset_dir /data/oak/ML_preconditioner_project/data_newA/MLCG_3D_newA_N64/b_rhs_90_10_V2 --output_dir /home/oak/projects/MLPCG_data/trained_models/data_V2/V101/ --input_matrix_A_dir /home/oak/projects/ML_preconditioner_project/MLCG_3D_newA_N64/data/output3d64_new_tgsl_rotating_fluid/matrixA_1.bin

import os, sys
import numpy as np
import torch
from torch.nn.functional import normalize
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data_dcdm")
sys.path.insert(1, os.path.join(dir_path, "lib"))
import conjugate_gradient as cg
import read_data as hf
from model import DCDM
from cg_tests import *

N = 64
DIM = 2
N2 = N**DIM
max_it = 50
tol = 1e-4
print("Loading model from disk.")
model = DCDM(DIM)
model_file = os.path.join(data_path, f"output_{DIM}D_{N}", "model_.pth")
model.load_state_dict(torch.load(model_file))
model.eval()
print("Loaded trained model from disk")


# testing data rhs
frame = 10
# test_smoke_path = os.path.join(dir_path, "..", "data_fluidnet", f"smoke_{DIM}D_{N}")
# b_smoke = hf.load_vector(os.path.join(test_smoke_path, f"div_v_star_{frame}.bin"), dtype='d').astype(np.float32)
# A_smoke = hf.readA_sparse(os.path.join(test_smoke_path, f"A_{frame}.bin"))
test_smoke_path = os.path.join(data_path, f"train_{DIM}D_{N}")

b = np.load(os.path.join(test_smoke_path, f"b_solid_{frame}.npy")).astype(np.float32)
b_t = torch.from_numpy(b)
A = hf.readA_sparse(os.path.join(test_smoke_path, "A_solid.bin"), dtype='f')

A_t = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, A.shape, dtype=torch.float32)

def dcdm_predict(r):
    global model
    with torch.no_grad():
        r = normalize(r, dim=0)
        r = r.view(*((1, 1,)+(N,)*DIM))
        x = model(r).flatten()
    return x
x_dcdm, res_t_history = dcdm(b_t, A_t, torch.zeros_like(b_t), dcdm_predict, max_it)

x_cg, res_cg = CG(b, A, np.zeros_like(b), max_it)

# print(x)
# print(residual)
