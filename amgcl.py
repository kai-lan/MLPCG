from cxx_src.build import pyamgcl
from cxx_src.build import pyamgcl_vexcl
from cxx_src.build import pyamgcl_cuda
import scipy.sparse as spa
import numpy as np
import sys
sys.path.append('lib')
from lib.read_data import *
import time
from scipy.io import mmread


frame = 200
path = f"{DATA_PATH}/ball_bowl_N256_200_3D"
A = readA_sparse(f"{path}/A_{frame}.bin")
rhs = load_vector(f"{path}/div_v_star_{frame}.bin")
flags = read_flags(f"{path}/flags_{frame}.bin")

# A_comp = compressedMat(A, flags)
# b_comp = compressedVec(rhs, flags)
A_comp = A
b_comp = rhs

# A_comp = readA_sparse("cxx_src/test_data/A_comp_999.bin")
# b_comp = load_vector("cxx_src/test_data/b_comp_999.bin")
# A_comp = mmread("cxx_src/test_data/poisson3Db/A.mtx")
# b_comp = mmread("cxx_src/test_data/poisson3Db/b.mtx")
# print(np.linalg.norm(b_comp))
x, info = pyamgcl_vexcl.solve(A_comp, b_comp, 1e-4, 1e-10, 100)
# start = time.time()
# for _ in range(10):
#     x, (iters, error) = amgcl_solver.solve(A_comp, b_comp)
# print("Total time", (time.time() - start) / 10)
print(info)
print("rel res", np.linalg.norm(b_comp - A_comp @ x) / np.linalg.norm(b_comp))
print("abs res", np.linalg.norm(b_comp - A_comp @ x))
# print(np.linalg.norm(x - y))
# amgcl_solver.print(A)