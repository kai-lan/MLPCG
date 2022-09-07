
project_name = "3D_256" #should change correct name
subproject_name = "SmokePlume"

trainedmodel_folder_name = "../dataset/train/trainedmodel"
data_folder_name = "../dataset/test/"+str(project_name)+"/"+str(subproject_name) + "/"
#MLV43_2_T2_3D_N64_ritz_vectors_20000_10000_10_90_json_E31
#MLV43_2_T2_3D_N128_ritz_vectors_20000_10000_10_90_json_E3

import sys
import os
import numpy as np
import tensorflow as tf
import gc
import scipy.sparse as sparse
from numpy.linalg import norm
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

dim = 256
dim2 = dim**3

# Input frame number as coomand variables
frame = int(sys.argv[1]) 

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

name_sparse_matrix = data_folder_name + "matrixA_"+str(frame)+".bin"   
A_sparse = hf.readA_sparse(dim, name_sparse_matrix,'f')
CG = cg.ConjugateGradientSparse(A_sparse)

if model_type == "F32":
    model = hf.load_model_from_source(trainedmodel_folder_name + "/" + project_name +"/saved_models/")
#here you can use float 16 type. converting code is convert.py
if model_type == "F16":
    model = hf.load_model_from_source(trainedmodel_folder_name + "/" + project_name +"/saved_models_float16/")


model.summary()
print("model has parameters is ",model.count_params())
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual

rhs_file = data_folder_name + "div_v_star"+str(frame)+".bin"
rhs = get_frame_from_source(rhs_file, d_type)
x_sol, res_arr= CG.cg_on_ML_generated_subspace(rhs, np.zeros(rhs.shape), model_predict, 10, tol, False ,True)

t0=time.time()                                                                                                                                                                                         
x_sol, res_arr= CG.cg_on_ML_generated_subspace(rhs, np.zeros(rhs.shape), model_predict, max_it_cg, tol, False ,True)
time_cg_ml = time.time() - t0
print("MLPCG ::::::::::: ", time_cg_ml," secs.")


p_out = "/results/"+project+"/frame_"+str(frame)
np.save(p_out, x_sol)
