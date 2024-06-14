'''
File: cg_tests.py
File Created: Wednesday, 8th February 2023 10:14:25 pm

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
from cxx_src.build import pyamgcl, pyamgcl_cuda, pyamgcl_vexcl
from cxx_src.build import pyic, pyic_cuda, pyic_vexcl
sys.path.append("cxx_src/pyamgx")
# import pyamgx
import time
from global_clock import GlobalClock

def npcg(b, A, x_init, model_predict, max_it, tol=1e-10, atol=1e-12, verbose=False, callback=None):
    timer = GlobalClock()
    timer.start('Total')
    timer.start('Init')

    norm_b = b.norm().item()
    r = b - A @ x_init
    norm = r.norm().item() / norm_b
    if callback:
        torch.cuda.synchronize()
        callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)

    if verbose:
        print(f"Iter {0}, residual {norm}, ares {r.norm().item()}")

    x = torch.clone(x_init)

    torch.cuda.synchronize()
    timer.stop('Init')

    timer.start('NN')
    z = model_predict(r, timer)
    torch.cuda.synchronize()
    timer.stop('NN')

    timer.start('CG')
    p = z
    w = A @ p
    rz = rz_old = r.dot(z)
    alpha = rz / p.dot(w)
    x += alpha * p
    r = b - A @ x
    torch.cuda.synchronize()
    timer.stop('CG')

    timer.start('Norm')
    norm = r.norm().item() / norm_b
    if callback:
        torch.cuda.synchronize()
        callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)
    torch.cuda.synchronize()
    timer.stop('Norm')

    k = 1
    while norm > max(tol, atol/norm_b) and k < max_it:
        timer.start('NN')
        z = model_predict(r, timer)
        torch.cuda.synchronize()
        timer.stop('NN')

        timer.start('CG')
        rz = r.dot(z)
        beta = rz / rz_old
        p = z + beta * p
        w = A @ p
        alpha = rz / p.dot(w)
        x += alpha * p
        r = b - A @ x
        k += 1
        rz_old = rz
        torch.cuda.synchronize()
        timer.stop('CG')

        timer.start('Norm')
        norm = r.norm().item() / norm_b
        torch.cuda.synchronize()
        timer.stop('Norm')
        if callback:
            torch.cuda.synchronize()
            callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)
        if verbose:
            print(f"Iter {k}, residual {norm}, ares {r.norm().item()}")

    torch.cuda.synchronize()
    timer.stop('Total')
    return x, k, timer, norm

def npcg_flex(b, A, x_init, model_predict, max_it, tol=1e-10, atol=1e-12, verbose=False, callback=None):
    timer = GlobalClock()

    timer.start('Total')
    timer.start('Init')

    norm_b = b.norm().item()
    r = b - A @ x_init
    norm = r.norm().item() / norm_b
    if callback:
        torch.cuda.synchronize()
        callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)

    if verbose:
        print(f"Iter {0}, residual {norm}, ares {r.norm().item()}")

    x = torch.clone(x_init)

    torch.cuda.synchronize()
    timer.stop('Init')

    timer.start('NN')
    z = model_predict(r, timer)
    torch.cuda.synchronize()
    timer.stop('NN')

    timer.start('CG')
    # z = r
    p = z
    w = A @ p
    rz = rz_old = r.dot(z)
    r_old = r
    alpha = rz / p.dot(w)
    x += alpha * p
    r = b - A @ x
    torch.cuda.synchronize()
    timer.stop('CG')

    timer.start('Norm')
    norm = r.norm().item() / norm_b
    if callback:
        torch.cuda.synchronize()
        callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)
    torch.cuda.synchronize()
    timer.stop('Norm')

    k = 1
    while norm > max(tol, atol/norm_b) and k < max_it:
        timer.start('NN')
        z = model_predict(r, timer)
        torch.cuda.synchronize()
        timer.stop('NN')

        timer.start('CG')
        # z = r
        rz = r.dot(z)
        beta = z.dot(r - r_old) / rz_old
        p = z + beta * p
        w = A @ p
        alpha = rz / p.dot(w)
        x += alpha * p
        r_old = r
        r = b - A @ x
        rz_old = rz
        k += 1
        torch.cuda.synchronize()
        timer.stop('CG')

        timer.start('Norm')
        norm = r.norm().item() / norm_b
        torch.cuda.synchronize()
        timer.stop('Norm')
        if callback:
            torch.cuda.synchronize()
            callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)
        if verbose:
            print(f"Iter {k}, residual {norm}, ares {r.norm().item()}")

    torch.cuda.synchronize()
    timer.stop('Total')
    return x, k, timer, norm


#####################################################################
# Our Neural-preconditioned steepest descent with orthogonalization
#####################################################################
def npsd(b, A, x_init, model_predict, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    timer = GlobalClock()

    timer.start('Total')
    timer.start('Init')

    norm_b = b.norm().item()
    r = b - A @ x_init
    norm = r.norm().item() / norm_b
    if callback:
        torch.cuda.synchronize()
        callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)

    if verbose:
        print(f"Iter {0}, residual {norm}, ares {r.norm().item()}")

    num_prev = 2
    p = []
    Ap = []
    alfa = []

    x_sol = torch.clone(x_init)

    imgs, c0, c1 = [], [], []

    torch.cuda.synchronize()
    timer.stop('Init')

    for i in range(1, max_it+1):

        timer.start('NN')
        q = model_predict(r, timer, imgs, c0, c1)

        torch.cuda.synchronize()
        timer.stop('NN')


        timer.start('Ortho')
        if num_prev > 0:
            if i > num_prev:
                for j in reversed(range(num_prev)):
                    q = q - q.dot(Ap[j])/alfa[j] * p[j]
                for j in range(num_prev-1):
                    p[j] = p[j+1]
                    Ap[j] = Ap[j+1]
                    alfa[j] = alfa[j+1]
            else:
                for j in reversed(range(i-1)):
                    q = q - q.dot(Ap[j])/alfa[j] * p[j]

        torch.cuda.synchronize()
        timer.stop('Ortho')

        timer.start('CG')
        if num_prev > 0:
            if i > num_prev:
                p[num_prev-1] = q
                Ap[num_prev-1] = A @ q
                alfa[num_prev-1] = p[-1].dot(Ap[-1])
            else:
                p.append(q)
                Ap.append(A @ q)
                alfa.append(p[-1].dot(Ap[-1]))

            beta = p[-1].dot(r)/alfa[-1]
        else:
            alpha = q.dot(A @ q)
            beta = q.dot(r)/alpha
        x_sol += q * beta


        r = b - A @ x_sol

        torch.cuda.synchronize()
        timer.stop('CG')

        timer.start('Norm')
        norm = r.norm().item() / norm_b
        if callback:
            torch.cuda.synchronize()
            callback(norm, time.perf_counter() - timer.top_level_clocks['Total'].start)

        if verbose:
            print(f"Iter {i}, residual {norm}, ares {r.norm().item()}")

        stop = norm < max(tol, atol/norm_b)
        torch.cuda.synchronize()
        timer.stop('Norm')

        if stop:
            torch.cuda.synchronize()
            timer.stop('Total')
            return x_sol, i, timer, norm
    torch.cuda.synchronize()
    timer.stop('Total')
    return x_sol, max_it, timer, norm


###################
# CG CPU
###################
def CG(b, A, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    tot_start = time.time()
    count = 0
    norm_b = np.linalg.norm(b)

    norm = np.linalg.norm(b - A @ x_init) / norm_b
    if callback:
        callback(norm.item(), time.time() - tot_start)
    if verbose:
        print(f"Iter {count}, residual {norm}")
    def res_callback(x):
        nonlocal count
        count += 1
        r = b - A @ x
        norm = np.linalg.norm(r) / norm_b
        if callback:
            callback(norm, time.time() - tot_start)
        if verbose:
            print(f"Iter {count}, residual {norm}")
    x, info = slin.cg(A, b, x0=x_init, tol=tol, atol=atol, maxiter=max_it, callback=res_callback)
    return x, count

###################
# CG CUDA
###################
def CG_GPU(b, A, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    tot_start = time.time()
    count = 0
    norm_b = cp.linalg.norm(b)
    norm = cp.linalg.norm(b - A @ x_init) / norm_b
    if callback:
        callback(norm.item(), time.time() - tot_start)
    if verbose:
        print(f"Iter {count}, residual {norm}")
    def res_callback(x):
        nonlocal count
        count += 1
        norm = cp.linalg.norm(b - A @ x) / norm_b
        if verbose:
            print(f"Iter {count}, residual {norm}")
        if callback:
            callback(norm.item(), time.time() - tot_start)
    x, info = gslin.cg(A, b, x0=x_init, tol=tol, atol=atol, maxiter=max_it, callback=res_callback)
    return x, count

def AMGX(b, A, x, sol, solver, max_it, tol=1e-10, atol=1e-10):
    t0 = time.time()
    solver.setup(A)
    solver.solve(b, x)
    t1 = time.time()
    x.download(sol)
    return sol, t1-t0

def AMGCL(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    x_comp, info = pyamgcl.solve(A_comp, b_comp, tol, atol, max_it, verbose)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def AMGCL_CUDA(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    x_comp, info = pyamgcl_cuda.solve(A_comp, b_comp, tol, atol, max_it, verbose)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def AMGCL_VEXCL(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    x_comp, info = pyamgcl_vexcl.solve(A_comp, b_comp, tol, atol, max_it, verbose)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def IC(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    x_comp, info = pyic.solve(A_comp, b_comp, tol, atol, max_it, verbose)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def IC_CUDA(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    x_comp, info = pyic_cuda.solve(A_comp, b_comp, tol, atol, max_it, verbose)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

def IC_VEXCL(b_comp, A_comp, x_init, max_it, tol=1e-10, atol=1e-10, verbose=False, callback=None):
    x_comp, info = pyic_vexcl.solve(A_comp, b_comp, tol, atol, max_it, verbose)
    iters, time, residual = info[0], info[1]+info[2], info[3]
    return x_comp, (iters, time, residual)

# https://scikit-sparse.readthedocs.io/en/latest/cholmod.html
def Cholesky_scikit_sparse(b, A,):
    from sksparse.cholmod import cholesky
    factor = cholesky(A, use_long=False)
    x = factor(b)
    return x

def Cholesky_cuda(b, A):
    from cholespy import CholeskySolverF, MatrixType # https://github.com/rgl-epfl/cholespy
    indptr = torch.tensor(A.indptr, device='cuda')
    indices = torch.tensor(A.indices, device='cuda')
    data = torch.tensor(A.data, device='cuda', dtype=torch.float32)
    b = torch.tensor(b, device='cuda', dtype=torch.float32)
    x = torch.zeros_like(b)
    solver = CholeskySolverF(A.shape[0], indptr, indices, data, MatrixType.CSC)
    solver.solve(b, x)
    return x.cpu().numpy()

def Cholesky_meshfem(b, A):
    sys.path.append(f"{dir_path}/../MeshFEM_dev/python")
    import sparse_matrices

    factorizer = sparse_matrices.CholeskyFactorizer(sparse_matrices.CholeskyProvider.CHOLMOD)
    A_mfem = sparse_matrices.SuiteSparseMatrix()
    A_mfem.m, A_mfem.n = A.shape
    A_mfem.Ai = A.indices
    A_mfem.Ap = A.indptr
    A_mfem.Ax = A.data
    A_mfem.nz = len(A.data)
    A_mfem.symmetry_mode = A_mfem.symmetry_mode.UPPER_TRIANGLE

    factorizer.factorize(A_mfem)

    x = factorizer.solve(b)
    return x

if __name__ == '__main__':
    import sys
    sys.path.append('lib')
    import time
    from lib.read_data import *
    N = 256
    frame = 102
    scene = 'waterflow_spiky_torus'
    data_path = "data"
    # data_path = "../tgsl/tgsl_projects/projects/incompressible_flow/build_3d"
    flags = read_flags(f"{data_path}/{scene}_N{N}_200_3D/flags_{frame}.bin")
    # sol = load_vector(f"{data_path}/{scene}_N{N}_200_3D/pressure_{frame}.bin")
    rhs = load_vector(f"{data_path}/{scene}_N{N}_200_3D/div_v_star_{frame}.bin")
    A = readA_sparse(f"{data_path}/{scene}_N{N}_200_3D/A_{frame}.bin", sparse_type='csc')

    A_lower = sparse.tril(A, format='csc')
    A_upper = sparse.triu(A, format='csc')
    # print(A_upper.shape, A_upper.nnz, A_upper.format)


    start = time.time()
    # x = Cholesky_meshfem(rhs, A_upper)
    # x = Cholesky_cuda(rhs, A_upper)
    # x = Cholesky_scikit_sparse(rhs, A_lower)
    print('Total time', time.time()-start)

    start = time.time()
    x = Cholesky_meshfem(rhs, A_upper)
    # x = Cholesky_cuda(rhs, A_upper)
    # x = Cholesky_scikit_sparse(rhs, A_lower)
    print('Total time', time.time()-start)


    r = rhs - A @ x
    print(x)
    # r = b.cpu().numpy() -  A @ x.cpu().numpy()
    norm = np.linalg.norm(r) / np.linalg.norm(rhs)
    print('Residual', norm)
    # print(A.shape)
    # rhs = load_vector("cxx_src/test_data/b_999.bin")
    # A = readA_sparse("cxx_src/test_data/A_999.bin")
    # rhs = compressedVec(rhs, flags)
    # A = compressedMat(A, flags)

    # start = time.time()
    # for _ in range(10):
    #     x, res = AMGCG(rhs.astype(np.float32), A.astype(np.float32), np.zeros_like(rhs).astype(np.float32), tol=1e-4, max_it=100, callback=None)
    # end = time.time()
    # print("Time:", (end-start)/10, "s.")

    # x, (iters, tot_time, res) = AMGCL(rhs, A, np.zeros_like(rhs), tol=1e-4, max_it=100, callback=None)
    # print("AMGCL CUDA Time:", tot_time, "s.")
    # x, (iters, tot_time, res) = AMGCL(rhs, A, np.zeros_like(rhs), tol=1e-4, max_it=100, callback=None)
    # print("AMGCL CUDA Time:", tot_time, "s.")
    # x, (iters, tot_time, res) = AMGCL(rhs, A, np.zeros_like(rhs), tol=1e-4, max_it=100, callback=None)
    # print("AMGCL CUDA Time:", tot_time, "s.")

    # start = time.time()
    # for _ in range(10):
    #     x, res = CG(rhs, A, np.zeros_like(rhs), max_it=1000, tol=1e-4, verbose=False)
    # end = time.time()
    # print("Time:", (end-start)/10, 's.')


    # rhs_cp, A_cp = cp.array(rhs, dtype=np.float64), cpsp.csr_matrix(A, dtype=np.float64)
    # start = time.time()
    # for _ in range(10):
    #     x_cg_cp, res_cg_cp = CG_GPU(rhs_cp, A_cp, cp.zeros_like(rhs_cp), 3000, tol=1e-4, verbose=False)
    # end = time.time()
    # print("CUDA CG Time:", (end-start)/10, "s.")


