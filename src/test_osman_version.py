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

import conjugate_gradient as cg
#import pressure_laplacian as pl
import helper_functions as hf

#%% Setup The Dimension and Load the Model
#Decide which dimention to test for:  64, 128, 256, 384, 512 (ToDo)
N = 128 # parser 1
#Decide which model to run: 64 or 128 and float type F16 (float 16) or F32 (float32)
# There are two types of models: k=64 and k=128, where the models trained over 
# the matrices ...
# k defines which parameters and model to be used. Currently we present two model.
# k = 64 uses model trained model
k = 64 # parser 2
float_type = "F16" #parser 3 (optional)
dataset_path = dir_path + "/../dataset"
trained_model_name = dataset_path+"/trained_models/model_N"+str(N)+"_from"+str(k)+"_"+float_type+"/"
model = hf.load_model_from_source(trained_model_name)
model.summary()
print("model has parameters is ",model.count_params())
if float_type == "F16":
    dtype_ = tf.float16
if float_type == "F32":
    dtype_ = tf.float32

# This is the lambda function that is needed in DGCM algorithm
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,N,N,N]),dtype=dtype_),training=False).numpy()[0,:,:].reshape([N**3]) #first_residual


#%% Load the matrix, vectors, and solver

#Decide which example to run for: SmokePlume, ...
#TODO: Add pictures to show different examples: SmokePlume, rotating_fluid
# old names: rotating_fluid -> output3d128_new_tgsl_rotating_fluid

example_name = "rotating_fluid" #parser
frame_number = 2 #parser
initial_normalization = False 
b_file_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "/" 
A_file_name = dataset_path + "/test_matrices_and_vectors/N"+str(N)+"/"+example_name + "/matrixA_"+str(frame_number)+".bin" 
b = hf.get_frame_from_source(frame_number, b_file_name, initial_normalization)
A = hf.readA_sparse(N, A_file_name,'f')
CG = cg.ConjugateGradientSparse(A)



#%%
# parameters for CG
tol = 1e-4 #parser
max_it_cg = 100
normalize_ = False 


#gpu_usage = 1024*48.0#int(1024*np.double(sys.argv[5]))
#which_gpu = 0#sys.argv[6]
#os.environ["CUDA_VISIBLE_DEVICES"]= ''
#gpus = tf.config.list_physical_devices('GPU')

#%% Testing
t0=time.time()                                                                                                                                                                                         
x_sol, res_arr= CG.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_predict, max_it_cg, tol, False ,True)
time_cg_ml = time.time() - t0
print("MLPCG ::::::::::: ", time_cg_ml," secs.")


#p_out = "/results/"+project+"/frame_"+str(frame)
#np.save(p_out, x_sol)

