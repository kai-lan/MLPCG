import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
import time
import argparse

import conjugate_gradient as cg
import helper_functions as hf

#os.environ["CUDA_VISIBLE_DEVICES"]= '' #not necessary

#%% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of the training matrix", default = 128)
parser.add_argument("-m", "--number_of_base_ritz_vectors", type=int,
                    help="number of ritz vectors to be used as the base for the dataset", default=10000)
parser.add_argument("--sample_size", type=int,
                    help="number of vectors to be created for dataset. I.e., size of the dataset", default=20000)
parser.add_argument("--theta", type=int,
                    help="see paper for the definition.", default=500)
parser.add_argument("--small_matmul_size", type=int,
                    help="Number of vectors.", default=200)
parser.add_argument("--dataset_dir", type=str,
                    help="path to the folder containing training matrix")
parser.add_argument("--output_dir", type=str,
                    help="path to the folder the training dataset to be saved")
args = parser.parse_args()

#%%
N = args.resolution

num_ritz_vectors = args.number_of_base_ritz_vectors

small_matmul_size = args.small_matmul_size

#save output_dir
import pathlib
pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True) 

#%% Load the matrix A
if N == 64:
    A_file_name = args.dataset_dir + "/original_matA/A_origN"+str(N)+".bin"  
elif N == 128:
    A_file_name = args.dataset_dir + "/original_matA/A_oriN"+str(N)+".bin"  
A = hf.readA_sparse(N, A_file_name,'f')
CG = cg.ConjugateGradientSparse(A)
rand_vec_x = np.random.normal(0,1, [N**3])
rand_vec = A.dot(rand_vec_x)
#%% Creating Lanczos Vectors:
print("Lanczos Iteration is running...")
W, diagonal, sub_diagonal = CG.lanczos_iteration_with_normalization_correction(rand_vec, num_ritz_vectors) #this can be loaded directly from c++ output
#W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_ritz_vectors, 1.0e-12) //Without ortogonalization. This is OK for 2D case.
print("Lanczos Iteration finished.")

#%% Create the tridiagonal matrix from diagonal and subdiagonal entries
tri_diag = np.zeros([num_ritz_vectors,num_ritz_vectors])
for i in range(1,num_ritz_vectors-1):
    tri_diag[i,i] = diagonal[i]
    tri_diag[i,i+1] = sub_diagonal[i]
    tri_diag[i,i-1]= sub_diagonal[i-1]
tri_diag[0,0]=diagonal[0]
tri_diag[0,1]=sub_diagonal[0]
tri_diag[num_ritz_vectors-1,num_ritz_vectors-1]=diagonal[num_ritz_vectors-1]
tri_diag[num_ritz_vectors-1,num_ritz_vectors-2]=sub_diagonal[num_ritz_vectors-2]


#%% Calculating eigenvectors of the tridiagonal matrix
print("Calculating eigenvectors of the tridiagonal matrix")
ritz_vals, Q0 = np.linalg.eigh(tri_diag)
ritz_vals = np.real(ritz_vals)
Q0 = np.real(Q0)
ritz_vectors = np.matmul(W.transpose(),Q0).transpose()


#%% For fast matrix multiply
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
b_rhs_temp = np.zeros([small_matmul_size,N**3])
cut_idx = int(num_ritz_vectors/2)+args.theta
num_zero_ritz_vals = 0
while ritz_vals[num_zero_ritz_vals] < 1.0e-8:
    num_zero_ritz_vals = num_zero_ritz_vals + 1
    
print("Creating Dataset ")
for it in range(0,for_outside):
    t0=time.time()
    sample_size = small_matmul_size
    coef_matrix = np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals,sample_size])
    coef_matrix[0:cut_idx] = 9*np.random.normal(0,1, [cut_idx,sample_size])
    l_b = small_matmul_size*it
    r_b = small_matmul_size*(it+1)
    #b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix).transpose()
    b_rhs_temp = mat_mult(ritz_vectors[num_zero_ritz_vals:num_ritz_vectors].transpose(),coef_matrix).transpose()

    for i in range(l_b,r_b):
        b_rhs_temp[i-l_b]=b_rhs_temp[i-l_b]/np.linalg.norm(b_rhs_temp[i-l_b])
        with open(args.output_dir+'/b_'+str(i)+'.npy', 'wb') as f:
            np.save(f, np.array(b_rhs_temp[i-l_b],dtype=np.float32))
print("Training Dataset is created.")














