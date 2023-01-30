#python3 train.py -N 64 --batch_size 1 --gpu_usage 40 --gpu_choice 0 --dataset_dir /data/oak/ML_preconditioner_project/data_newA/MLCG_3D_newA_N64/b_rhs_90_10_V2 --output_dir /home/oak/projects/MLPCG_data/trained_models/data_V2/V101/ --input_matrix_A_dir /home/oak/projects/ML_preconditioner_project/MLCG_3D_newA_N64/data/output3d64_new_tgsl_rotating_fluid/matrixA_1.bin

import os, sys
import numpy as np
import torch
from torch.nn.functional import normalize
dir_path = os.path.dirname(os.path.relpath(__file__))
data_path = os.path.join(dir_path, "..", "data_dcdm")
sys.path.insert(1, os.path.join(dir_path, "..", "lib"))
import conjugate_gradient as cg
import read_data as hf
from train import DCDM

N = 64
DIM = 3
N2 = N**DIM
max_it = 30
tol = 1e-4
print("Loading model from disk.")
model = DCDM(DIM)
model_file = os.path.join(data_path, f"output_{DIM}D_{N}", "model_Mon-Jan--9-01:52:28-2023.pth")
model.load_state_dict(torch.load(model_file))
model.eval()
print("Loaded trained model from disk")


# testing data rhs
# rand_vec_x = np.random.normal(0,1, [N2])
# b_rand = CG.multiply_A_sparse(rand_vec_x)

test_smoke_path = os.path.join(data_path, "test_matrices_and_vectors", "N64", "smoke_passing_bunny")
b_smoke = hf.load_vector(os.path.join(test_smoke_path, "div_v_star10.bin"), dtype='d').astype(np.float32)
A_smoke = hf.readA_sparse(64, os.path.join(test_smoke_path, "matrixA_10.bin"), DIM=3)

test_rotate_path = os.path.join(data_path, "test_matrices_and_vectors", "N128", "rotating_fluid")
b_rotate = hf.load_vector(os.path.join(test_rotate_path, "div_v_star_2.bin"))

def model_predict(r):
    global model
    with torch.no_grad():
        r = torch.tensor(r, dtype=torch.float32)
        r = normalize(r, dim=0)
        r = r.view(*((1, 1,)+(N,)*DIM))
        x = model(r).flatten().numpy()
    return x
# model_predict = lambda r: model(normalize(torch.tensor(r, dtype=torch.float32), dim=0).view(*((1, 1,)+(N,)*DIM))).flatten().detach().numpy()
CG_smoke = cg.ConjugateGradientSparse(A_smoke)
x, res_arr_ml_generated_cg = CG_smoke.dcdm(b_smoke, np.zeros(b_smoke.shape), model_predict, max_it,tol, True)
residual = np.linalg.norm(b_smoke - A_smoke @ x) / np.linalg.norm(b_smoke)
print(x)
print(residual)
