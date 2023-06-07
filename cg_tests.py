'''
File: cg_tests.py
File Created: Wednesday, 8th February 2023 10:14:25 pm

Author: Kai Lan (kai.weixian.lan@gmail.com)
--------------
'''
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(dir_path, "lib"))

import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse.linalg as slin
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as gslin # https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html#module-cupyx.scipy.sparse.linalg
import pyamg # https://github.com/pyamg/pyamg
import matplotlib.pyplot as plt
from lib.read_data import compute_weight

###################
# FluidNet / DCDM
###################
def dcdm(b, A, x_init, model_predict, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    N = math.isqrt(len(b))
    assert N**2 == len(b), "RHS vector dimension is incorrect"
    norm_b = b.norm().item()
    r = b - A @ x_init
    # np.save(f"res_0.npy", r.cpu().numpy())
    # print('tol', tol, 'atol/normb', norm_b)
    if norm_type == 'l2': norm = r.norm().item() / norm_b
    else: norm = x_init.dot(A @ x_init).item() / 2 - x_init.dot(b)
    res_history = [norm]
    if verbose:
        print(f"Iter {0}, residual {res_history[-1]}, ares {r.norm().item()}")

    num_prev = 2
    p = []
    Ap = []
    alfa = []
    for _ in range(num_prev):
        p.append(torch.zeros_like(b))
        Ap.append(torch.zeros_like(b))
        alfa.append(1.0)

    x_sol = torch.clone(x_init)
    for i in range(1, max_it+1):
        q = model_predict(r) # r_normalized =r / norm_r. approximate A^-1 r
        # a1, a0 = q.dot(Ap1)/alpha1, q.dot(Ap0)/alpha0
        # print(a1, a0)
        for j in reversed(range(num_prev)):
            q = q - q.dot(Ap[j])/alfa[j] * p[j]

        for j in range(num_prev-1):
            p[j] = p[j+1]
            Ap[j] = Ap[j+1]
            alfa[j] = alfa[j+1]

        p[num_prev-1] = q
        Ap[num_prev-1] = A @ q
        alfa[num_prev-1] = p[-1].dot(Ap[-1])

        beta = p[-1].dot(r)/alfa[-1]

        x_sol += p[-1] * beta
        r = b - A @ x_sol
        # np.save(f"res_{i}.npy", r.cpu().numpy())
        if norm_type == 'l2': norm = r.norm().item() / norm_b
        else: norm = x_sol.dot(A @ x_sol).item() / 2 - x_sol.dot(b)
        res_history.append(norm)
        if verbose:
            print(f"Iter {i}, residual {res_history[-1]}, ares {r.norm().item()}")
        if norm < max(tol, atol/norm_b):
            return x_sol, res_history
    return x_sol, res_history
###################
# CG
###################
def CG(b, A, x_init, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    count = 0
    norm_b = np.linalg.norm(b)
    # np.save(f"res_0.npy", b)
    if norm_type == 'l2': norm = np.linalg.norm(b - A @ x_init) / norm_b
    else: norm = x_init.dot(A @ x_init) / 2 - x_init.dot(b)
    res_history = [norm]
    if verbose:
        print(f"Iter {count}, residual {res_history[-1]}")
    def res_callback(x):
        nonlocal count
        count += 1
        r = b - A @ x
        # np.save(f"res_{count}.npy", r)
        if norm_type == 'l2': norm = np.linalg.norm(b - A @ x) / norm_b
        else: norm = x.dot(A @ x) / 2 - x.dot(b)
        res_history.append(norm)
        if verbose:
            print(f"Iter {count}, residual {res_history[-1]}")
    x, info = slin.cg(A, b, x0=x_init, tol=tol, atol=atol, maxiter=max_it, callback=res_callback)
    return x, res_history

###################
# CG on GPU
###################
def CG_GPU(b, A, x_init, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    count = 0
    norm_b = cp.linalg.norm(b)
    # print('norm_b', norm_b)
    if norm_type == 'l2': norm = cp.linalg.norm(b - A @ x_init) / norm_b
    else: norm = x_init.dot(A @ x_init) / 2 - x_init.dot(b)
    res_history = [norm.item()]
    if verbose:
        print(f"Iter {count}, residual {res_history[-1]}")
    def callback(x):
        nonlocal count
        count += 1
        if norm_type == 'l2': norm = cp.linalg.norm(b - A @ x) / norm_b
        else: norm = x.dot(A @ x) / 2 - x.dot(b)
        res_history.append(norm.item())
        if verbose:
            print(f"Iter {count}, residual {res_history[-1]}")
    x, info = gslin.cg(A, b, x0=x_init, tol=tol, atol=atol, maxiter=max_it, callback=callback)
    return x, res_history

def AMGCG(b, A, x_init, max_it, tol=1e-10, atol=1e-12, callback=None):
    ml = pyamg.ruge_stuben_solver(A)
    residuals = []
    x = ml.solve(b, x0=x_init, maxiter=max_it, tol=tol, residuals=residuals, callback=callback)
    b_norm = np.linalg.norm(b)
    for i in range(len(residuals)):
        residuals[i] /= b_norm
    return x, residuals

if __name__ == '__main__':
    import sys
    sys.path.append('lib')
    from lib.read_data import *
    N = 1024
    frame = 1
    scene = 'dambreak'
    flags = read_flags(f"data/{scene}_N{N}_200/flags_{frame}.bin")
    fluids = np.argwhere(flags == 2)
    air = np.argwhere(flags == 3)

    sol = load_vector(f"data/{scene}_N{N}_200/pressure_{frame}.bin")
    rhs = load_vector(f"data/{scene}_N{N}_200/div_v_star_{frame}.bin")
    A = readA_sparse(f"data/{scene}_N{N}_200/A_{frame}.bin")

    # rhs = compressedVec(rhs, flags)
    # A = compressedMat(A, flags)
    # rr = []
    # def callback(x):
    #     r = rhs - A @ x
    #     rr.append(r)
    #     print(np.linalg.norm(r))

    x, res = AMGCG(rhs.astype(np.float32), A.astype(np.float32), np.zeros_like(rhs).astype(np.float32), tol=1e-5, max_it=100, callback=None)
    for i, r in enumerate(res):
        print(i, r)
    # x, res = CG(rhs.astype(np.float32), A.astype(np.float32), np.zeros_like(rhs).astype(np.float32), max_it=100, tol=1e-4, verbose=True)
    # rhs_cp, A_cp = cp.array(rhs, dtype=np.float64), cpsp.csr_matrix(A, dtype=np.float64)
    # x_cg_cp, res_cg_cp = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), 3000, tol=1e-4, verbose=True)

    # r = np.empty((361, A.shape[0]), dtype=np.float64)
    # for i in range(860):
    #     r = np.load(f"res_{i}.npy")
    #     torch.save(torch.tensor(r, dtype=torch.float32), f"data/{scene}_N{N}_200/preprocessed/{frame}/b_{i}_res.pt")
    # coeff =np.random.rand(400, 361)
    # for i in range(361):
    #     coeff[:, i] *= 361- i
    # rr = coeff @ r
    # for i in range(400):
        # rr[i] /= np.linalg.norm(rr[i])
        # torch.save(torch.tensor(rr[i], dtype=torch.float32), f"data/{scene}_N{N}_200/preprocessed/{frame}/b_{i}_res.pt")
    # for i in range(80, 90):
    #     r_i = np.load(f"res_{i}.npy")
    #     for j in range(i-10, i):
    #         r_j = np.load(f"res_{j}.npy")
    #         dot = r_i.dot(r_j)
    #         print(i, j, dot)
