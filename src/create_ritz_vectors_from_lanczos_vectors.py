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
parser.add_argument("--lanczos_dir", type=str,
                    help="path to the folder containing lanczoz vector")
parser.add_argument("--output_dir", type=str,
                    help="path to the folder the ritz vectors to be saved")
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
rand_vec_x = np.random.normal(0,1, [N**3])
rand_vec = A.dot(rand_vec_x)


#%%
print("Loading Lanczos Vectors")
W = np.zeros([num_ritz_vectors,N**3])

for i in range(num_ritz_vectors):
    if i%1000==0:
        print(i)
    lv_filename = args.lanczos_dir+"/"+str(i)+".bin"
    W[i] = hf.load_vector(lv_filename)

diagonal = np.zeros([num_ritz_vectors])
sub_diagonal = np.zeros([num_ritz_vectors-1])
for i in range(num_ritz_vectors-1):
    Av = A.dot(W[i])
    diagonal[i] = np.dot(W[i],Av)
    sub_diagonal[i] = np.dot(Av,W[i+1])

Av = A.dot(W[num_ritz_vectors-1])
diagonal[num_ritz_vectors-1] = np.dot(W[num_ritz_vectors-1],Av)


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
print("Creating and saving ritz vectors...")
ritz_vectors_temp = np.zeros([100,N**3])
#%%
num_for_loops = int(num_ritz_vectors/100)
for i in range(100):
    print("i = ",i)
    ll = 100*i
    rr = ll+100
    print(ll)
    #% %
    #ritz_vectors[ll:rr] = np.matmul(W.transpose(),Q0[:,ll:rr]).transpose()
    ritz_vectors_temp = np.matmul(W.transpose(),Q0[:,ll:rr]).transpose()

    for i in range(ll,rr):
        #with open(project_data_folder3+'ritz_vectors_10000_3D_N'+str(dim-1)+'/'+str(i)+'.npy', 'wb') as f:
        with open(args.output_dir+'/'+str(i)+'.npy', 'wb') as f:
            np.save(f, np.array(ritz_vectors_temp[i-ll],dtype=np.float32))












