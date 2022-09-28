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
                    help="number of ritz vectors to be used as the base in the dataset", default=10000)
parser.add_argument("--sample_size", type=int,
                    help="number of vectors to be created for dataset", default=20000)
parser.add_argument("--theta", type=int,
                    help="see paper for.", default=500)
parser.add_argument("--small_matmul_size", type=int,
                    help="Number of vectors.", default=200)
parser.add_argument("--dataset_dir", type=str,
                    help="path to the folder containing training matrix", default="/data/oak/dataset_mlpcg")
parser.add_argument("--output_dir", type=str,
                    help="folder that saves the dataset")
args = parser.parse_args()

#%%
N = args.resolution

num_ritz_vectors = args.number_of_base_ritz_vectors

small_matmul_size = args.small_matmul_size

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

print("testing")
i = 60
j = 60
print("i = ",i,", j = ",j)
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(eigvals[i])
i = 60
j = 90
print("i = ",i,", j = ",j)
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(eigvals[i])

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
while eigvals[num_zero_ritz_vals] < 1.0e-8:
    num_zero_ritz_vals = num_zero_ritz_vals + 1
    
print(num_zero_ritz_vals)

print(" Creating Rhs's")
for it in range(0,for_outside):
    t0=time.time()
    sample_size = small_matmul_size
    coef_matrix = np.random.normal(0,1, [num_ritz_vectors-1,sample_size])
    coef_matrix[0:cut_idx] = 9*np.random.normal(0,1, [cut_idx,sample_size])

    l_b = small_matmul_size*it
    r_b = small_matmul_size*(it+1)
    print(it)
    #b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix[:,l_b:r_b]).transpose()
    #b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix).transpose()
    b_rhs_temp = mat_mult(ritz_vectors[1:num_ritz_vectors].transpose(),coef_matrix).transpose()

    #% % Making sure b is in the range of A
    for i in range(l_b,r_b):
        if i%10 == 0:
            print(i)
        #b_rhs[0] = b_rhs[0] - sum(b_rhs[0][reduced_idx])/len(reduced_idx)
        #b_rhs[0][zero_idxs]=0
        b_rhs_temp[i-l_b]=b_rhs_temp[i-l_b]/np.linalg.norm(b_rhs_temp[i-l_b])
        #with open(project_data_folder3+'b_rhs_20000_20000_ritz_vectors_V2_for_3D_random_N'+str(dim-1)+'/'+str(i)+'.npy', 'wb') as f:
        with open(args.output_dir+'/b_'+str(i)+'.npy', 'wb') as f:
            np.save(f, np.array(b_rhs_temp[i-l_b],dtype=np.float32))
        
    print(norm(b_rhs_temp)**2)
    time_cg_ml = int((time.time() - t0))
    print("data creation at ",it, " took ", time_cg_ml, " seconds.")

















