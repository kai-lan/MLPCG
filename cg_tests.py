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
import cupyx.scipy.sparse.linalg as gslin # https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html#module-cupyx.scipy.sparse.linalg
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
        print(f"Iter {0}, residual {res_history[-1]}")
    p0 = torch.zeros_like(b)
    p1 = torch.zeros_like(b)
    Ap0 = torch.zeros_like(b)
    Ap1 = torch.zeros_like(b)
    alpha0, alpha1 = 1.0, 1.0
    x_sol = torch.clone(x_init)
    for i in range(1, max_it+1):
        q = model_predict(r) # r_normalized =r / norm_r. approximate A^-1 r
        # a1, a0 = q.dot(Ap1)/alpha1, q.dot(Ap0)/alpha0
        # print(a1, a0)
        q = q - q.dot(Ap1)/alpha1 * p1 - q.dot(Ap0)/alpha0 * p0
        p0, p1 = p1, q
        Ap0, Ap1 = Ap1, A @ q
        alpha0, alpha1 = alpha1, p1.dot(Ap1)
        beta = p1.dot(r)/alpha1
        x_sol += p1 * beta
        r = b - A @ x_sol
        # np.save(f"res_{i}.npy", r.cpu().numpy())
        if norm_type == 'l2': norm = r.norm().item() / norm_b
        else: norm = x_sol.dot(A @ x_sol).item() / 2 - x_sol.dot(b)
        res_history.append(norm)
        if verbose:
            print(f"Iter {i}, residual {res_history[-1]}")
        if norm < max(tol, atol/norm_b):
            return x_sol, res_history
    return x_sol, res_history
###################
# CG
###################
def CG(b, A, x_init, max_it, tol=1e-10, atol=1e-12, verbose=False, norm_type='l2'):
    count = 0
    norm_b = np.linalg.norm(b)
    if norm_type == 'l2': norm = np.linalg.norm(b - A @ x_init) / norm_b
    else: norm = x_init.dot(A @ x_init) / 2 - x_init.dot(b)
    res_history = [norm]
    if verbose:
        print(f"Iter {count}, residual {res_history[-1]}")
    def callback(x):
        nonlocal count
        count += 1
        r = b - A @ x
        # np.save(f"res_{count}.npy", r)
        if norm_type == 'l2': norm = np.linalg.norm(b - A @ x) / norm_b
        else: norm = x.dot(A @ x) / 2 - x.dot(b)
        res_history.append(norm)
        if verbose:
            print(f"Iter {count}, residual {res_history[-1]}")
    x, info = slin.cg(A, b, x0=x_init, tol=tol, atol=atol, maxiter=max_it, callback=callback)
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
