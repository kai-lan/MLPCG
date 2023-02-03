import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(dir_path, "..", "lib"))

import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
import time
from train import FluidNet
import read_data as hf

torch.set_grad_enabled(False) # disable autograd globally

###################
# FluidNet
###################
def fluidnet_dcdm(b, A, flags, x_init, model, max_it, tol=1e-10):
    N = math.isqrt(len(b))
    assert N**2 == len(b), "RHS vector dimension is incorrect"
    r = b - A @ x_init
    norm_r = r.norm().item()
    print(f"Iter {0}, residual {norm_r}")
    p0 = torch.zeros_like(b)
    p1 = torch.zeros_like(b)
    Ap0 = torch.zeros_like(b)
    Ap1 = torch.zeros_like(b)
    alpha0, alpha1 = 1.0, 1.0
    x_sol = torch.clone(x_init)
    for i in range(max_it):
        q = model(torch.stack([r, flags]).view(1, 2, N, N)).flatten() # r_normalized =r / norm_r. approximate A^-1 r
        q -= q.dot(Ap1)/alpha1 * p1 + q.dot(Ap0)/alpha0 * p0
        p0, p1 = p1, q
        Ap0, Ap1 = Ap1, A @ q
        alpha0, alpha1 = alpha1, p1.dot(Ap1)
        beta = p1.dot(r)/alpha1
        x_sol += p1 * beta
        r = b - A @ x_sol
        norm_r = r.norm().item()
        print(f"Iter {i+1}, residual {norm_r}")
        if norm_r < tol:
            print("DCDM converged in ", i+1, " iterations to residual ", norm_r)
            return x_sol
    return x_sol
###################
# CG
###################
def CG(b, A, x_init, max_it, tol=1e-10):
    count = 0
    norm_r = np.linalg.norm(b - A @ x_init)
    print(f"Iter {count}, residual {norm_r}")
    def callback(x):
        nonlocal count
        count += 1
        norm_r = np.linalg.norm(b - A @ x)
        print(f"Iter {count}, residual {norm_r}")
    x, info = slin.cg(A, b, x0=x_init, tol=tol, maxiter=max_it, callback=callback)
    assert info > 0, "Illegal input or breakdown"
    return x

if __name__ == "__main__":
    N = 64
    DIM = 2
    frame = 600
    max_cg_iter = 3
    test_dambreak_path = os.path.join(dir_path, "..", "data_fluidnet", f"dambreak_{DIM}D_{N}")
    model_dambreak_path = os.path.join(dir_path, "..", "data_fluidnet", f"output_{DIM}D_{N}")

    print(f"{'-'*40}\nFluidnet\n{'-'*40}")
    model = FluidNet()
    model_file = os.path.join(model_dambreak_path, "model_Thu-Feb--2-21:24:25-2023.pth")
    model.load_state_dict(torch.load(model_file))
    model.eval()




    A = torch.load(os.path.join(test_dambreak_path, "preprocessed", f"A_{frame}.pt"))
    flags  = torch.tensor(hf.read_flags(os.path.join(test_dambreak_path, f"flags_{frame}.bin")), dtype=torch.float32)
    b = torch.tensor(hf.load_vector(os.path.join(test_dambreak_path, f"div_v_star_{frame}.bin")), dtype=torch.float32)
    # x_gt = torch.from_numpy(hf.load_vector(os.path.join(test_dambreak_path, f"pressure_{frame}.bin"), dtype='d').astype(np.float32))
    x = fluidnet_dcdm(b, A, flags, torch.zeros_like(b), model, max_it=max_cg_iter)

    b_dambreak = hf.load_vector(os.path.join(test_dambreak_path, f"div_v_star_{frame}.bin"), dtype='d').astype(np.float32)
    A_dambreak = hf.readA_sparse(64, os.path.join(test_dambreak_path, f"A_{frame}.bin"), DIM=2).astype(np.float32)
    print(f"{'-'*40}\nCG\n{'-'*40}")
    x = CG(b_dambreak, A_dambreak, np.zeros_like(b_dambreak), max_it=max_cg_iter)
