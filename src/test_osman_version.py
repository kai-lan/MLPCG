#Loading the Required Libraries
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
#Decide which dimention to test for:  64, 128, 256, 384, 512(ToDo)
N = 128
N3 = N**3 
#Decide which model to run: 64 or 128 and float type F16 (float 16) or F32 (float32)
# k defines which parameters and model to be used. Currently we present two model.
# k=64 uses model trained model
k = 64
float_type = "F16"
trained_model_name = dir_path+"/../dataset/trained_models/model_N"+str(N)+"_from"+str(k)+"_"+float_type+"/"

model = hf.load_model_from_source(trained_model_name)

model.summary()
print("model has parameters is ",model.count_params())
if float_type == "F16":
    dtype_ = tf.float16
if float_type == "F32":
    dtype_ = tf.float32
    
    
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,N,N,N]),dtype=dtype_),training=False).numpy()[0,:,:].reshape([N3]) #first_residual


#%% Decide which example to Run, and load the matrix and vectors
#Decide which example to run for: SmokePlume, ...
#TODO: Add pictures to show different examples  
example_name = "SmokePlume"
# Input frame number as coomand variables
frame_number = int(sys.argv[1]) 
#data_folder_name = "../dataset/test/"+str(project_name)+"/"+str(subproject_name) + "/" #Change this part

data_folder_name = project_folder_general+"../MLCG_3D_newA_N128/data/output3d128_new_tgsl_rotating_fluid/"
b_smoke = hf.get_frame_from_source(b_rhs_n, data_folder_name, initial_normalization)

#%%



# parameters for CG
tol = 1e-4 
max_it_cg = 50
normalize_ = False 
model_type = "F32"


#gpu_usage = 1024*48.0#int(1024*np.double(sys.argv[5]))
#which_gpu = 0#sys.argv[6]
#os.environ["CUDA_VISIBLE_DEVICES"]= str(which_gpu)
#gpus = tf.config.list_physical_devices('GPU')


#%% Getting RHS for the Testing
d_type='double'
def get_frame_from_source(file_rhs,d_type='double'):
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        r1 = np.delete(r0, [0])
        return r1            

name_sparse_matrix = data_folder_name + "matrixA_"+str(frame_number)+".bin"   
A_sparse = hf.readA_sparse(dim, name_sparse_matrix,'f')
CG = cg.ConjugateGradientSparse(A_sparse)


rhs_file = data_folder_name + "div_v_star"+str(frame)+".bin"
rhs = get_frame_from_source(rhs_file, d_type)
x_sol, res_arr= CG.cg_on_ML_generated_subspace(rhs, np.zeros(rhs.shape), model_predict, 10, tol, False ,True)

t0=time.time()                                                                                                                                                                                         
x_sol, res_arr= CG.cg_on_ML_generated_subspace(rhs, np.zeros(rhs.shape), model_predict, max_it_cg, tol, False ,True)
time_cg_ml = time.time() - t0
print("MLPCG ::::::::::: ", time_cg_ml," secs.")


p_out = "/results/"+project+"/frame_"+str(frame)
np.save(p_out, x_sol)

