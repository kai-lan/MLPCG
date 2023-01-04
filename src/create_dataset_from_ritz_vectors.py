import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np
# import tensorflow as tf
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
                    help="N or resolution of the training matrix", default = 64)
parser.add_argument("-m", "--number_of_base_ritz_vectors", type=int,
                    help="number of ritz vectors to be used as the base for the dataset", default=10000)
parser.add_argument("--sample_size", type=int,
                    help="number of vectors to be created for dataset. I.e., size of the dataset", default=20000)
parser.add_argument("--theta", type=int,
                    help="see paper for the definition.", default=500)
parser.add_argument("--small_matmul_size", type=int,
                    help="Number of vectors.", default=200)
parser.add_argument("--ritz_dir", type=str,
                    help="path to the folder containing Ritz vectors")
parser.add_argument("--output_dir", type=str,
                    help="path to the folder the dataset vectors to be saved")
parser.add_argument("--input_matrix_A_dir", type=str,
                    help="path to the matrix_A file. It should be .bin file")
args = parser.parse_args()

#%%
N = args.resolution

num_ritz_vectors = args.number_of_base_ritz_vectors

small_matmul_size = args.small_matmul_size

os.makedirs(args.output_dir, exist_ok=True)

#%%
A_file_name = args.input_matrix_A_dir
A = hf.readA_sparse(N, A_file_name,'f')
CG = cg.ConjugateGradientSparse(A)


#%% Saving and loading ritz values
ritz_vectors = np.zeros([num_ritz_vectors,N**3])
print("Loading Ritz Vectors")
for i in range(num_ritz_vectors):
    if i%1000 ==0:
        print(i)
    with open(args.ritz_dir+'/'+str(i)+'.npy', 'rb') as f:
        ritz_vectors[i] = np.load(f)

# test ritz vals
ritz_values_last20 = CG.create_ritz_values(ritz_vectors[num_ritz_vectors-20:num_ritz_vectors])
ritz_values_first20 = CG.create_ritz_values(ritz_vectors[0:20])
print("last 20: ",ritz_values_last20)
print("first 20: ",ritz_values_first20)

#%% fast parallel matrix multiply
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
#m, n, c = 1000, 1500, 1200
#A = np.random.randint(1, 50, size = (m, n))
#B = np.random.randint(1, 50, size = (n, c))
#res = mat_mult(A, B)


#%%
import time
print("numba")

small_size = 200
b_rhs_temp = np.zeros([small_size,N**3])
cut_idx = int(num_ritz_vectors/2)+args.theta
num_zero_ritz_vals = 1
sample_size = 20000

print(" Creating Rhs's")
for it in range(0,100):
    t0=time.time()
    sample_size = small_size
    coef_matrix = np.random.normal(0,1, [num_ritz_vectors-1,sample_size])
    coef_matrix[0:cut_idx] = 9*np.random.normal(0,1, [cut_idx,sample_size])

    l_b = small_size*it
    r_b = small_size*(it+1)
    print(it)
    #b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix[:,l_b:r_b]).transpose()
    #b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix).transpose()
    b_rhs_temp = mat_mult(ritz_vectors[1:num_ritz_vectors].transpose(),coef_matrix).transpose()

    #% % Making sure b is in the range of A
    for i in range(l_b,r_b):
        if i%10 == 0:
            print(i)
        b_rhs_temp[i-l_b]=b_rhs_temp[i-l_b]/np.linalg.norm(b_rhs_temp[i-l_b])
        with open(args.output_dir+'/'+str(i)+'.npy', 'wb') as f:
            np.save(f, np.array(b_rhs_temp[i-l_b],dtype=np.float32))

    print(norm(b_rhs_temp)**2)
    time_cg_ml = int((time.time() - t0))
    print("data creation at ",it, " took ", time_cg_ml, " seconds.")








