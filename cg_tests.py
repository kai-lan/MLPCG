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
from cxx_src.build import pyamgcl
from cxx_src.build import pyamgcl_cuda
from cxx_src.build import pyamgcl_vexcl
import time

###################
# DCDM
###################
def dcdm(b, A, x_init, model_predict, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    tot_time = {"Total": 0.0, "NN": 0.0, "Ortho": 0.0, "CG": 0.0, "Norm": 0.0, "Init": 0.0}

    tot_start = time.time()
    start = time.time()

    norm_b = b.norm().item()
    r = b - A @ x_init
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

    tot_time['Init'] += time.time() - start

    for i in range(1, max_it+1):
        start = time.time()

        q = model_predict(r)

        tot_time['NN'] += time.time() - start

        start = time.time()

        for j in reversed(range(num_prev)):
            q = q - q.dot(Ap[j])/alfa[j] * p[j]
        for j in range(num_prev-1):
            p[j] = p[j+1]
            Ap[j] = Ap[j+1]
            alfa[j] = alfa[j+1]

        tot_time['Ortho'] += time.time() - start

        start = time.time()

        p[num_prev-1] = q
        Ap[num_prev-1] = A @ q
        alfa[num_prev-1] = p[-1].dot(Ap[-1])
        beta = p[-1].dot(r)/alfa[-1]
        x_sol += p[-1] * beta
        r = b - A @ x_sol

        tot_time['CG'] += time.time() - start

        start = time.time()

        if norm_type == 'l2': norm = r.norm().item() / norm_b
        else: norm = x_sol.dot(A @ x_sol).item() / 2 - x_sol.dot(b)
        res_history.append(norm)

        tot_time['Norm'] += time.time() - start

        if verbose:
            print(f"Iter {i}, residual {res_history[-1]}, ares {r.norm().item()}")
        if norm < max(tol, atol/norm_b):
            tot_time['Total'] += time.time() - tot_start
            return x_sol, res_history, tot_time
    return x_sol, res_history, tot_time

###################
# CG CPU
###################
def CG(b, A, x_init, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    count = 0
    norm_b = np.linalg.norm(b)

    if norm_type == 'l2': norm = np.linalg.norm(b - A @ x_init) / norm_b
    else: norm = x_init.dot(A @ x_init) / 2 - x_init.dot(b)
    res_history = [norm]
    if verbose:
        print(f"Iter {count}, residual {res_history[-1]}")
    def res_callback(x):
        nonlocal count
        count += 1
        r = b - A @ x

        if norm_type == 'l2': norm = np.linalg.norm(b - A @ x) / norm_b
        else: norm = x.dot(A @ x) / 2 - x.dot(b)
        res_history.append(norm)
        if verbose:
            print(f"Iter {count}, residual {res_history[-1]}")
    x, info = slin.cg(A, b, x0=x_init, tol=tol, atol=atol, maxiter=max_it, callback=res_callback)
    return x, res_history

###################
# CG CUDA
###################
def CG_GPU(b, A, x_init, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    count = 0
    norm_b = cp.linalg.norm(b)

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

def AMGCL(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-12, callback=None):
    x_comp, info = pyamgcl.solve(A_comp, b_comp, tol, atol, max_it)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def AMGCL_CUDA(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-12, callback=None):
    x_comp, info = pyamgcl_cuda.solve(A_comp, b_comp, tol, atol, max_it)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def AMGCL_VEXCL(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-12, callback=None):
    x_comp, info = pyamgcl_vexcl.solve(A_comp, b_comp, tol, atol, max_it)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

if __name__ == '__main__':
    import sys
    sys.path.append('lib')
    import time
    from lib.read_data import *
    N = 256
    frame = 200
    scene = 'ball_bowl'
    data_path = "data"
    # data_path = "../tgsl/tgsl_projects/projects/incompressible_flow/build_3d"
    flags = read_flags(f"{data_path}/{scene}_N{N}_200_3D/flags_{frame}.bin")
    sol = load_vector(f"{data_path}/{scene}_N{N}_200_3D/pressure_{frame}.bin")
    rhs = load_vector(f"{data_path}/{scene}_N{N}_200_3D/div_v_star_{frame}.bin")
    A = readA_sparse(f"{data_path}/{scene}_N{N}_200_3D/A_{frame}.bin")


    # rhs = load_vector("cxx_src/test_data/b_999.bin")
    # A = readA_sparse("cxx_src/test_data/A_999.bin")
    # rhs = compressedVec(rhs, flags)
    # A = compressedMat(A, flags)

    # start = time.time()
    # for _ in range(10):
    #     x, res = AMGCG(rhs.astype(np.float32), A.astype(np.float32), np.zeros_like(rhs).astype(np.float32), tol=1e-4, max_it=100, callback=None)
    # end = time.time()
    # print("Time:", (end-start)/10, "s.")

    # start = time.time()
    # for _ in range(10):
    x, (iters, tot_time, res) = AMGCL_CUDA(rhs, A, np.zeros_like(rhs), tol=1e-4, max_it=100, callback=None)
    # end = time.time()
    print("AMGCL CUDA Time:", tot_time, "s.")

    # start = time.time()
    # for _ in range(10):
    #     x, res = CG(rhs, A, np.zeros_like(rhs), max_it=1000, tol=1e-4, verbose=False)
    # end = time.time()
    # print("Time:", (end-start)/10, 's.')


    rhs_cp, A_cp = cp.array(rhs, dtype=np.float64), cpsp.csr_matrix(A, dtype=np.float64)
    start = time.time()
    for _ in range(10):
        x_cg_cp, res_cg_cp = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), 3000, tol=1e-4, verbose=False)
    end = time.time()
    print("CUDA CG Time:", (end-start)/10, "s.")


