from cxx_src.build import pyamgcl_ext
import scipy.sparse as spa
import numpy as np
import sys
sys.path.append('lib')
from lib.read_data import *

solver_config = pyamgcl_ext.SolverConfig()
solver_config.tol = 1e-4
solver_config.max_iter = 100
amgcl_solver = pyamgcl_ext.AMGCLSolver(solver_config)

path = f"{DATA_PATH}/dambreak_N128_200_3D"
A = readA_sparse(f"{path}/A_10.bin")
rhs = load_vector(f"{path}/div_v_star_10.bin")
flags = read_flags(f"{path}/flags_10.bin")

A_comp = compressedMat(A, flags)
b_comp = compressedVec(rhs, flags)

x, (iters, error) = amgcl_solver.solve(A_comp, b_comp)

print(iters, error)
print(np.linalg.norm(b_comp - A_comp @ x) / np.linalg.norm(b_comp))
# print(np.linalg.norm(x - y))
# amgcl_solver.print(A)