#python3 train.py -N 64 --batch_size 1 --gpu_usage 40 --gpu_choice 0 --dataset_dir /data/oak/ML_preconditioner_project/data_newA/MLCG_3D_newA_N64/b_rhs_90_10_V2 --output_dir /home/oak/projects/MLPCG_data/trained_models/data_V2/V101/ --input_matrix_A_dir /home/oak/projects/ML_preconditioner_project/MLCG_3D_newA_N64/data/output3d64_new_tgsl_rotating_fluid/matrixA_1.bin 

import os
import sys
import numpy as np
from tensorflow import keras

import tensorflow as tf 
import gc
import scipy.sparse as sparse
import time
#import matplotlib.pyplot as plt
import argparse

sys.path.insert(1, '../lib/')
import conjugate_gradient as cg
import helper_functions as hf

#%% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of the training matrix", default = 64)

parser.add_argument("--gpu_usage", type=int,
                    help="gpu usage, in terms of GB.", default=10)
parser.add_argument("--gpu_choice", type=str,
                    help="which gpu to use.", default='0')
parser.add_argument("--model_dir", type=str,
                    help="path to the folder containing model ")
parser.add_argument("--test_frame", type=int,
                    help="frame number", default=10)
parser.add_argument("--test1_dir", type=str,
                    help="path to the folder containing test 1. Must include rhs and matrix files (.bin format)",default='')
parser.add_argument("--test2_dir", type=str,
                    help="path to the folder containing test 2",default='')
parser.add_argument("--output_dir", type=str,
                    help="saving residuals. Only saves test1",default='') 
parser.add_argument("--max_it", type=int,
                    help="max_it", default=30)
parser.add_argument("--tol", type=float,
                    help="tolerance", default=1.0e-4)


args = parser.parse_args()
#%%
N = args.resolution
N2 = N**3
max_it = args.max_it
tol = args.tol

# you can modify gpu memory usage editing here
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_choice
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1024*np.double(args.gpu_usage))])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


print("Loading model from disk.")
json_file = open(args.model_dir + '/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(args.model_dir + "/model.h5")
print("Loaded trained model from disk") 

model.summary() 


#%% testing data rhs

#rand_vec_x = np.random.normal(0,1, [N2])
#b_rand = CG.multiply_A_sparse(rand_vec_x)

#data_folder_name = project_folder_general+"data/output3d64_smoke/"
#b_smoke = hf.get_frame_from_source(10, data_folder_name)

#data_folder_name = project_folder_general+"data/output3d128_smoke_sigma/"
#b_rotate = hf.get_frame_from_source(10, data_folder_name)


model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,N,N,N]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([N2]) #first_residual


#print("Smoke Plume Test")
#x_sol, res_arr_ml_generated_cg = CG.dcdm(b_smoke, np.zeros(b.shape), model_predict, max_it,tol, True)
#print("Random RHS Test")
#x_sol, res_arr_ml_generated_cg = CG.dcdm(b_rand, np.zeros(b_rand.shape), model_predict, max_it,tol, True)

if args.test1_dir != '':
    print("Test 1")
    #get rhs b
    b1 = hf.get_frame_from_source(args.test_frame, args.test1_dir)
    A_sparse_scipy_test1 = hf.readA_sparse(N, args.test1_dir+"/matrixA_"+str(args.test_frame)+'.bin','f')
    CG_test1 = cg.ConjugateGradientSparse(A_sparse_scipy_test1)
    #get test matrix:
    x_sol, res_arr_ml_generated_cg = CG_test1.dcdm(b1, np.zeros(b1.shape), model_predict, max_it,tol, True)    
if args.test2_dir != '':
    b2 = hf.get_frame_from_source(args.test_frame, args.test2_dir)




    
