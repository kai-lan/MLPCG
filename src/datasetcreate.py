import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np
import scipy
from numpy.linalg import norm
import time
import argparse

import conjugate_gradient as cg
import read_data as hf

#os.environ["CUDA_VISIBLE_DEVICES"]= '' #not necessary

#%% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of the training matrix", default = 64)
parser.add_argument("-m", "--number_of_base_ritz_vectors", type=int,
                    help="number of ritz vectors to be used as the base for the dataset", default=1000)
parser.add_argument("--sample_size", type=int,
                    help="number of vectors to be created for dataset. I.e., size of the dataset", default=2000)
parser.add_argument("--theta", type=int,
                    help="see paper for the definition.", default=50) # j < m/2 + theta low frequency spectrum
parser.add_argument("--small_matmul_size", type=int,
                    help="Number of vectors.", default=200)
parser.add_argument("--dataset_dir", type=str,
                    help="path to the folder containing training matrix")
parser.add_argument("--output_dir", type=str,
                    help="path to the folder the training dataset to be saved")
args = parser.parse_args()

N = args.resolution
DIM = 2
outdir = "dataset_mlpcg/train_64_2D"
bc = "air"
num_ritz_vectors = args.number_of_base_ritz_vectors
small_matmul_size = args.small_matmul_size

#save output_dir
os.makedirs(outdir, exist_ok=True)

# Load the matrix A
A_file_name = f"dataset_mlpcg/train_64_2D/A_{bc}.bin"
A = hf.readA_sparse(N, A_file_name, DIM, dtype='f')
CG = cg.ConjugateGradientSparse(A)
rand_vec_x = np.random.normal(0,1, [N**DIM])
rand_vec = A.dot(rand_vec_x)
# Creating Lanczos Vectors:
print("Lanczos Iteration is running...")
W, diagonal, sub_diagonal = CG.lanczos_iteration_with_normalization_correction(rand_vec, num_ritz_vectors) #this can be loaded directly from c++ output
#W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_ritz_vectors, 1.0e-12) //Without ortogonalization. This is OK for 2D case.
print("Lanczos Iteration finished.")

# Calculating eigenvectors of the tridiagonal matrix: only return eigvals > 1e-8
print("Calculating eigenvectors of the tridiagonal matrix")
ritz_vals, Q0 = scipy.linalg.eigh_tridiagonal(diagonal, sub_diagonal, select='v', select_range=(1.0e-8, np.inf))
ritz_vectors = np.matmul(W.transpose(), Q0).transpose()

# For fast matrix multiply
from numba import njit, prange
@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res

for_outside = int(args.sample_size/small_matmul_size)
b_rhs_temp = np.zeros([small_matmul_size, N**3])
cut_idx = int(num_ritz_vectors/2) + args.theta

print("Creating Dataset ")
for it in range(0,for_outside):
    t0=time.time()
    sample_size = small_matmul_size
    coef_matrix = np.random.normal(0,1, [len(ritz_vals), sample_size])
    coef_matrix[0:cut_idx] = 9*np.random.normal(0,1, [cut_idx, sample_size])
    l_b = small_matmul_size*it
    r_b = small_matmul_size*(it+1)
    #b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix).transpose()
    b_rhs_temp = coef_matrix.T @ ritz_vectors #mat_mult(ritz_vectors.transpose(), coef_matrix).transpose()

    for i in range(l_b,r_b):
        b_rhs_temp[i-l_b] = b_rhs_temp[i-l_b]/np.linalg.norm(b_rhs_temp[i-l_b])
        with open(f"{outdir}/b_{bc}_{i}.npy", 'wb') as f:
            np.save(f, np.array(b_rhs_temp[i-l_b],dtype=np.float32))
print("Training Dataset is created.")














