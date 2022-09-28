# This is an example test code for the paper 
# The test code solves the linear system A x = b, where A is pressure matrix, and b is the velocity divergence
# Both A and b is creted from a simulation. We provide various simulations, such as smoke plume, rotating fluid, etc, 
# for reader to pick to test.
# A and b also depends on the frame number

#%% Load the required libraries
import sys
import os    
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np 
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
import time
import argparse

import conjugate_gradient as cg
#import pressure_laplacian as pl
#import helper_functions as hf

#this makes sure that we are on cpu
os.environ["CUDA_VISIBLE_DEVICES"]= ''


#%% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128, 256, 384],
                    help="N or resolution of test", default = 128)
parser.add_argument("-k", "--trained_model_type", type=int, choices=[64, 128],
                    help="which model to test", default=128)
parser.add_argument("-f", "--float_type", type=int, choices=[16, 32],
                    help="model parameters' float type", default=32)
parser.add_argument("-ex", "--example_type", type=str, choices=["rotating_fluid", "smoke_passing_bunny"],
                    help="example type", default="smoke_passing_bunny")
parser.add_argument("-fn", "--frame_number", type=int,
                    help="example type", default=10)
parser.add_argument("--max_cg_iter", type=int,
                    help="maximum cg iteration", default=1000)
parser.add_argument("-tol","--tolerance", type=float,
                    help="tolerance for both DGCM and CG algorithm", default=1.0e-4)
parser.add_argument("--verbose_dgcm", type=bool,
                    help="prints residuals of DGCM algorithm for each iteration", default=False)
parser.add_argument('--skip_dcdm', dest='skip_dcdm', action="store_false", 
                    const=True, default=False, help='skips dcdm tests')
#action='store_const',

args = parser.parse_args()
#%%
N = args.resolution

k = args.trained_model_type

float_type = args.float_type
if float_type == 16:
    dtype_ = tf.float16
if float_type == 32:
    dtype_ = tf.float32

example_name = args.example_type

frame_number = args.frame_number

max_cg_iter = args.max_cg_iter

tol = args.tolerance 

verbose_dgcm = args.verbose_dgcm

verbose_ldlt = verbose_dgcm

skip_dcdm = args.skip_dcdm

print("skip_dcdm ", skip_dcdm)

#%% 
# if matrix does not change in the example, use the matrix for the first frame.  
if example_name in ["rotating_fluid", "smoke_passing_bunny"]:
    matrix_frame_number = 1
else:
    matrix_frame_number = frame_number
    

#%% Setup The Dimension and Load the Model
#Decide which dimention to test for:  64, 128, 256, 384, 512 (ToDo)
#N = 128 # parser 1
#Decide which model to run: 64 or 128 and float type F16 (float 16) or F32 (float32)
# There are two types of models: k=64 and k=128, where the models trained over 
# the matrices ...
# k defines which parameters and model to be used. Currently we present two model.
# k = 64 uses model trained model
#dataset_path = "/home/ayano/project/tgsl/projects/incompressible_flow/build/"
dataset_path = "/data/oak/dataset_mlpcg"
#"/data/oak/dataset_mlpcg" # change this to where you put the dataset folder
trained_model_name = dataset_path + "/trained_models/model_N"+str(N)+"_from"+str(k)+"_F"+str(float_type)+"/"
model = hf.load_model_from_source(trained_model_name)

model.summary()
print("number of parameters in the model is ",model.count_params())

# This is the lambda function that is needed in DGCM algorithm
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,N,N,N]),dtype=dtype_),training=False).numpy()[0,:,:].reshape([N**3]) #first_residual


#%% Load the matrix, vectors, and solver

#Decide which example to run for: SmokePlume, ...
#TODO: Add pictures to show different examples: SmokePlume, rotating_fluid
# old names: rotating_fluid -> output3d128_new_tgsl_rotating_fluid

#%% Getting RHS for the Testing
d_type='double'
def get_vector_from_source(file_rhs,d_type='double'):
    if(os.path.exists(file_rhs)):
        return_vector = np.fromfile(file_rhs, dtype=d_type)
        return_vector = np.delete(return_vector, [0])
        return return_vector
    else:
        print("RHS does not exist at "+ file_rhs)
        


initial_normalization = False 
b_file_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "/div_v_star"+str(matrix_frame_number)+".bin" 
A_file_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+ example_name +"/matrixA_"+str(matrix_frame_number)+".bin" 
b = get_vector_from_source(b_file_name)
A = hf.readA_sparse(N, A_file_name,'f')
CG = cg.ConjugateGradientSparse(A)


#%%
# parameters for CG
normalize_ = False 


#gpu_usage = 1024*48.0#int(1024*np.double(sys.argv[5]))
#which_gpu = 0#sys.argv[6]
#gpus = tf.config.list_physical_devices('GPU')

#%% Testing
# Dummy Calling:
model_predict(b)

print("DGCM is running...")
t0=time.time()                                  
max_DGCM_iter = 100                                                                                                                                      
x_sol, res_arr= CG.DGCM(b, np.zeros(b.shape), model_predict, max_DGCM_iter, tol, False ,verbose_dgcm)
time_cg_ml = time.time() - t0
print("DGCM took ", time_cg_ml," secs.")


print("CG is running...")
t0=time.time()
x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,max_cg_iter,tol,True)
time_cg = time.time() - t0
print("CG took ",time_cg, " secs")

## deflated version of cg ##


#t0=time.time()
#x_sol_cg, res_arr_cg = CG.deflated_pcg(b, max_cg_iter, tol, 13, True)
#(self, b, max_outer_it = 100, pcg_inner_it = 1, tol = 1.0e-15, method = "jacobi", num_vectors = 16, verbose = False):   
#time_cg = time.time() - t0
#print("Deflated PCG took ",time_cg, " secs")


print("LDLT is running...")

t0=time.time()
L, D = CG.ldlt()
time_ldlt_creation = time.time() - t0
print("L and D computed in ", time_ldlt_creation, " seconds.")
l_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "_matrix_L_fn"+str(matrix_frame_number)+".npz" 
d_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "_matrix_D_fn"+str(matrix_frame_number)+".npz" 
sparse.save_npz(l_name, L)
sparse.save_npz(d_name, D)
#L = sparse.load_npz(l_name)
#D = sparse.load_npz(d_name)

print("LDLT PCG is running...")
t0 = time.time()
x,res_arr_cg  = CG.ldlt_pcg(L, D, b, max_cg_iter, tol, verbose_ldlt)
time_ldlt_pcg = time.time() - t0
print("LDLT PCG took ", time_ldlt_pcg, " seconds.")

#p_out = "/results/"+project+"/frame_"+str(frame)
#np.save(p_out, x_sol)

