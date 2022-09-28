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
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128, 256, 384],
                    help="N or resolution of the training matrix", default = 128)
parser.add_argument("-m", "--number_of_base_ritz_vectors", type=int,
                    help="number of ritz vectors to be used as the base in the dataset", default=10000)
parser.add_argument("--dataset_dir", type=str,
                    help="path to the dataset", default="/data/oak/dataset_mlpcg")
args = parser.parse_args()

#%%
N = args.resolution

num_ritz_vectors = args.number_of_base_ritz_vectors

#%% Load the matrix A
A_file_name = args.dataset_dir + "/test_matrices_and_vectors/N"+str(N) + "/matrixA_orig.bin"  #Change this one tho origional matrix (important)
A = hf.readA_sparse(N, A_file_name,'f')
CG = cg.ConjugateGradientSparse(A)
rand_vec_x = np.random.normal(0,1, [N**3])
rand_vec = A.dot(rand_vec_x)
#this could be the original name
#name_sparse_matrix = project_folder_general + "data/output3d"+str(dim)+"_new_tgsl_rotating_fluid/matrixA_"+str(1)+".bin"   


#%% Creating Lanczos Vectors:
W, diagonal, sub_diagonal = CG.lanczos_iteration_with_normalization_correction(rand_vec, num_ritz_vectors) #this can be loaded directly from c++ output
#W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_ritz_vectors, 1.0e-12) //Without ortogonalization. This is OK for 2D case.

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
eigvals, Q0 = np.linalg.eigh(tri_diag)
eigvals = np.real(eigvals)
Q0 = np.real(Q0)
#%%
#ritz_vectors = np.zeros(W.shape)
ritz_vectors = np.matmul(W.transpose(),Q0).transpose()
print(eigvals[0:10], eigvals[-10:-1])



