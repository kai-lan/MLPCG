#python3 train.py -N 64 --batch_size 1 --gpu_usage 40 --gpu_choice 0 --dataset_dir /data/oak/ML_preconditioner_project/data_newA/MLCG_3D_newA_N64/b_rhs_90_10_V2 --output_dir /home/oak/projects/MLPCG_data/trained_models/data_V2/V101/ --input_matrix_A_dir /home/oak/projects/ML_preconditioner_project/MLCG_3D_newA_N64/data/output3d64_new_tgsl_rotating_fluid/matrixA_1.bin 

import os
import sys
import numpy as np
from tensorflow import keras

import tensorflow as tf 
#import gc
#import scipy.sparse as sparse
import time
#import matplotlib.pyplot as plt
import argparse

sys.path.insert(1, '../lib/')
import conjugate_gradient as cg
import helper_functions as hf
import get_model

#%% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of the training matrix", default = 64)
parser.add_argument("--sample_size", type=int,
                    help="number of vectors to be created for dataset. I.e., size of the dataset", default=20000)
parser.add_argument("--batch_size", type=int,
                    help="--batch_size.", default=200)
parser.add_argument("--total_number_of_epochs", type=int,
                    help="Total number of epochs for training", default=1000)
parser.add_argument("--epoch_save_period", type=int,
                    help="Represents epoch save periodicity", default=1)
#parser.add_argument("--loading_number", type=int,
#                    help="Number of vectors per inner step training.", default=1000)
parser.add_argument("--inner_loop_total", type=int,
                    help="number loops in inner training", default=20)
parser.add_argument("--gpu_usage", type=int,
                    help="gpu usage, in terms of GB.", default=3)
parser.add_argument("--start_epoch", type=int,
                    help="if saved, start from the saved epoch.", default=0)
parser.add_argument("--gpu_choice", type=str,
                    help="which gpu to use.", default='0')
parser.add_argument("--dataset_dir", type=str,
                    help="path to the folder containing dataset vectors")
parser.add_argument("--num_training_vectors", type=int,
                    help="number of vectors in trea",default=20000)
parser.add_argument("--test_frame", type=int,
                    help="frame number", default=10)
parser.add_argument("--test1_dir", type=str,
                    help="path to the folder containing test 1",default='')
parser.add_argument("--test2_dir", type=str,
                    help="path to the folder containing test 2",default='')
parser.add_argument("--output_dir", type=str,
                    help="save_folder for models")
parser.add_argument("--input_matrix_A_dir", type=str,
                    help="path to the matrix_A file. It should be .bin file")
parser.add_argument("--model_type", type=str,
                    help="model type. Look get_model.py")

args = parser.parse_args()
#%%
N = args.resolution
N2 = N**3
Models = get_model.get_model(N)
lr = 1.0e-4  # learning rate


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


#%% Load the training matrix and correcponding loss function
A_sparse_scipy = hf.readA_sparse(N, args.input_matrix_A_dir,'f')
CG = cg.ConjugateGradientSparse(A_sparse_scipy)
coo = A_sparse_scipy.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
A_sparse = tf.SparseTensor(indices, np.float32(coo.data), coo.shape)

def custom_loss_function_cnn_1d_fast(y_true,y_pred):
    b_size_ = len(y_true)
    err = 0
    for i in range(args.batch_size):
        A_tilde_inv = 1/tf.tensordot(tf.reshape(y_pred[i],[1,N2]), tf.sparse.sparse_dense_matmul(A_sparse, tf.reshape(y_pred[i],[N2,1])),axes=1)
        qTb = tf.tensordot(tf.reshape(y_pred[i],[1,N2]), tf.reshape(y_true[i],[N2,1]), axes=1)
        x_initial_guesses = tf.reshape(y_pred[i],[N2,1]) * qTb * A_tilde_inv
        err = err + tf.reduce_sum(tf.math.square(tf.reshape(y_true[i],[N2,1]) - tf.sparse.sparse_dense_matmul(A_sparse, x_initial_guesses)))
    return err/b_size_


#%% Set model and losses
training_loss_name = args.output_dir + "/training_loss.npy"
validation_loss_name =  args.output_dir +"/validation_loss.npy"

if args.start_epoch == 0:
    print("Creating model "+args.model_type)
    model = Models.get_predefined_model(args.model_type)
    print("Untrained model created.")
    training_loss = []
    validation_loss = []
else:
    print("Loading model from disk. Starting from epoch = ",args.start_epoch)
    model_name = args.output_dir + '/'+ str(args.start_epoch)
    json_file = open(model_name + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_name + "/model.h5")
    print("Loaded trained model from disk") 
        
    with open(training_loss_name, 'rb') as f:
        training_loss = list(np.load(f))
    with open(validation_loss_name, 'rb') as f:
        validation_loss = list(np.load(f))
        
    # to not lose data
    training_loss_name_old = args.output_dir + "/training_loss_old.npy"
    validation_loss_name_old =  args.output_dir +"/validation_loss_old.npy"
    with open(training_loss_name_old, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name_old, 'wb') as f:
        np.save(f, np.array(validation_loss))

    training_loss = training_loss[0:args.start_epoch]
    validation_loss = validation_loss[0:args.start_epoch]
    print("training_loss so far \n",training_loss)
    print("validation_loss so far \n", validation_loss)

    
model.compile(optimizer="Adam", loss=custom_loss_function_cnn_1d_fast) 
model.optimizer.lr = lr;
model.summary() 


#%% testing data rhs

rand_vec_x = np.random.normal(0,1, [N2])
b_rand = CG.multiply_A_sparse(rand_vec_x)

#data_folder_name = project_folder_general+"data/output3d64_smoke/"
#b_smoke = hf.get_frame_from_source(10, data_folder_name)

#data_folder_name = project_folder_general+"data/output3d128_smoke_sigma/"
#b_rotate = hf.get_frame_from_source(10, data_folder_name)



#%%
loading_number = round(args.total_data_points/args.inner_loop_total)
#for_loading_number = round(total_data_points/loading_number)
b_rhs = np.zeros([loading_number,N2])
#perm = np.random.permutation(total_data_points)
with open(args.dataset_dir+'/perm.npy', 'rb') as f:  
    perm = np.load(f)

for i in range(1,args.total_number_of_epochs):    
    print("Training at i = " + str(i+args.start_epoch))
    
    training_loss_inner = []
    validation_loss_inner = []
    t0=time.time()    
    for ii in range(args.inner_loop_total):
        print("Sub_training at ",ii,"/",args.inner_loop_total," at training ",i)

    
        for j in range(loading_number):
            with open(args.dataset_dir+'/'+str(perm[loading_number*ii+j])+'.npy', 'rb') as f:  
                b_rhs[j] = np.load(f)
        
        sub_train_size = round(0.9*loading_number)
        sub_test_size = loading_number - sub_train_size
        iiln = ii*loading_number
        x_train = tf.convert_to_tensor(b_rhs[0:loading_number].reshape([loading_number,N,N,N,1]),dtype=tf.float32) 
        x_test = tf.convert_to_tensor(b_rhs[sub_train_size:loading_number].reshape([sub_test_size,N,N,N,1]),dtype=tf.float32)         
         
        hist = model.fit(x_train,x_train,
                        epochs=args.epoch_save_period,
                        batch_size=args.batch_size,
                        shuffle=True,
                        validation_data=(x_test,x_test))
        
        training_loss_inner = training_loss_inner + hist.history['loss']
        validation_loss_inner = validation_loss_inner + hist.history['val_loss']  
    
    time_cg_ml = (time.time() - t0)
    print("Training loss at i = ",sum(training_loss_inner)/args.inner_loop_total)
    print("Validation loss at i = ",sum(training_loss_inner)/args.inner_loop_total)
    print("Time for epoch = ",i," is ", time_cg_ml)
    training_loss = training_loss + [sum(validation_loss_inner)/args.inner_loop_total]
    validation_loss = validation_loss + [sum(validation_loss_inner)/args.inner_loop_total]
    
    model_json_dir = args.output_dir+"/"+str(args.start_epoch +args.epoch_save_period*i)
    os.system("mkdir "+model_json_dir)
    os.system("touch "+model_json_dir+"/model.json")
    model_json = model.to_json()

    with open(model_json_dir+ "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_json_dir + "/model.h5")
    
    with open(training_loss_name, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name, 'wb') as f:
        np.save(f, np.array(validation_loss))
    print(training_loss)
    print(validation_loss)
    
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,N,N,N]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([N2]) #first_residual
    max_it=30
    tol=1.0e-12

    #print("Smoke Plume Test")
    #x_sol, res_arr_ml_generated_cg = CG.dcdm(b_smoke, np.zeros(b.shape), model_predict, max_it,tol, True)
    print("Random RHS Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_rand, np.zeros(b_rand.shape), model_predict, max_it,tol, True)

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


