# https://github.com/pyamg/pyamg
import sys
sys.path.append('lib')
import pyamg
import numpy as np
from lib.read_data import *
A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid

N = 512
frame = 1
path = f"data/wedge_N{N}_200"
A = readA_sparse(os.path.join(path, f"A_{frame}.bin"))
b = load_vector(os.path.join(path, f"div_v_star_{frame}.bin"))
# flags = read_flags(os.path.join(path, f"flags_{frame}.bin"))

ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
print(ml)                                           # print hierarchy information
# b = np.random.rand(A.shape[0])
residuals = []
x = ml.solve(b, tol=1e-5, residuals=residuals)                          # solve Ax=b to a tolerance of 1e-10
print("residual: ", np.linalg.norm(b-A*x)/np.linalg.norm(b))          # compute norm of residual vector

for i in range(len(residuals)):
    print(i, residuals[i]/np.linalg.norm(b))