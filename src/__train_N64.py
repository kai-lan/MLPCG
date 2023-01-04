import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf 
import gc
import scipy.sparse as sparse
import time
#import matplotlib.pyplot as plt

project_name = "3D_N64"
project_folder_subname = os.path.basename(os.getcwd())
print("project_folder_subname = ", project_folder_subname)
project_folder_general = "../dataset/train/forTraining/3D_N64"

sys.path.insert(1, '../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

dim = 64
dim2 = dim**3
lr = 1.0e-4


# command variables
epoch_num = int(sys.argv[1])
epoch_each_iter = int(sys.argv[2])
b_size = int(sys.argv[3])
loading_number = int(sys.argv[4])


# you can modify gpu memory usage editing here

gpu_usage = int(1024*np.double(sys.argv[5]))
which_gpu = sys.argv[6]

os.environ["CUDA_VISIBLE_DEVICES"]=which_gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=gpu_usage)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


name_sparse_matrix = project_folder_general+"/matrixA.bin"
A_sparse_scipy = hf.readA_sparse(dim, name_sparse_matrix,'f')

CG = cg.ConjugateGradientSparse(A_sparse_scipy)

coo = A_sparse_scipy.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
A_sparse = tf.SparseTensor(indices, np.float32(coo.data), coo.shape)

def custom_loss_function_cnn_1d_fast(y_true,y_pred):
    b_size_ = len(y_true)
    err = 0
    for i in range(b_size):
        A_tilde_inv = 1/tf.tensordot(tf.reshape(y_pred[i],[1,dim2]), tf.sparse.sparse_dense_matmul(A_sparse, tf.reshape(y_pred[i],[dim2,1])),axes=1)
        qTb = tf.tensordot(tf.reshape(y_pred[i],[1,dim2]), tf.reshape(y_true[i],[dim2,1]), axes=1)
        x_initial_guesses = tf.reshape(y_pred[i],[dim2,1]) * qTb * A_tilde_inv
        err = err + tf.reduce_sum(tf.math.square(tf.reshape(y_true[i],[dim2,1]) - tf.sparse.sparse_dense_matmul(A_sparse, x_initial_guesses)))
    return err/b_size_

#%% Training model 
fil_num=16
input_rhs = keras.Input(shape=(dim, dim, dim, 1))
first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

apa = layers.AveragePooling3D((2, 2,2), padding='same')(lb) 
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa

upa = layers.UpSampling3D((2, 2,2))(apa) + lb
upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa
upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

last_layer = layers.Dense(1, activation='linear')(upa)

model = keras.Model(input_rhs, last_layer)
model.compile(optimizer="Adam", loss=custom_loss_function_cnn_1d_fast) 
model.optimizer.lr = lr;
model.summary()

#%% testing data rhs

rand_vec_x = np.random.normal(0,1, [dim2])
b_rand = CG.multiply_A_sparse(rand_vec_x)

data_folder_name = project_folder_general+"data/output3d64_smoke/"
b_smoke = hf.get_frame_from_source(10, data_folder_name)

data_folder_name = project_folder_general+"data/output3d128_smoke_sigma/"
b_rotate = hf.get_frame_from_source(10, data_folder_name)


#%%
training_loss_name = project_folder_general+project_folder_subname+"/"+project_name+"_training_loss.npy"
validation_loss_name = project_folder_general+project_folder_subname+"/"+project_name+"_validation_loss.npy"
training_loss = []
validation_loss = []

d_name = "b_rhs_20000_10000_ritz_vectors_newA_90_10_random_N64"

#%%
total_data_points = 20000
for_loading_number = round(total_data_points/loading_number)
b_rhs = np.zeros([loading_number,dim2])

for i in range(1,epoch_num):    
    print("Training at i = " + str(i))
    
    training_loss_inner = []
    validation_loss_inner = []
    t0=time.time()    
    perm = np.random.permutation(total_data_points)
    for ii in range(for_loading_number):
        print("Sub_training at ",ii,"/",for_loading_number," at training ",i)

        
        #d_name = "b_rhs_10000_eigvector_equidistributed_random_N"
        # Loasing the data
        for j in range(loading_number):
            with open(foldername+str(perm[loading_number*ii+j])+'.npy', 'rb') as f:  
                b_rhs[j] = np.load(f)
        
        sub_train_size = round(0.9*loading_number)
        sub_test_size = loading_number - sub_train_size
        iiln = ii*loading_number
        x_train = tf.convert_to_tensor(b_rhs[0:loading_number].reshape([loading_number,dim,dim,dim,1]),dtype=tf.float32) 
        x_test = tf.convert_to_tensor(b_rhs[sub_train_size:loading_number].reshape([sub_test_size,dim,dim,dim,1]),dtype=tf.float32)         
         
        hist = model.fit(x_train,x_train,
                        epochs=epoch_each_iter,
                        batch_size=b_size,
                        shuffle=True,
                        validation_data=(x_test,x_test))
        
        training_loss_inner = training_loss_inner + hist.history['loss']
        validation_loss_inner = validation_loss_inner + hist.history['val_loss']  
    
    time_cg_ml = (time.time() - t0)
    print("Training loss at i = ",sum(training_loss_inner)/for_loading_number)
    print("Validation loss at i = ",sum(training_loss_inner)/for_loading_number)
    print("Time for epoch = ",i," is ", time_cg_ml)
    training_loss = training_loss + [sum(validation_loss_inner)/for_loading_number]
    validation_loss = validation_loss + [sum(validation_loss_inner)/for_loading_number]
    
    os.system("mkdir ./saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i))
    os.system("touch ./saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/model.json")
    model_json = model.to_json()
    model_name_json = project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/"
    with open(model_name_json+ "model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name_json + "model.h5")
    
    with open(training_loss_name, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name, 'wb') as f:
        np.save(f, np.array(validation_loss))
    print(training_loss)
    print(validation_loss)
    
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
    max_it=30
    tol=1.0e-12

    print("Rotating Fluid Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_rotate, np.zeros(b_rotate.shape), model_predict, max_it,tol, True)    
    print("Smoke Plume Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_smoke, np.zeros(b.shape), model_predict, max_it,tol, True)
    print("RandomRHSi Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_rand, np.zeros(b_rand.shape), model_predict, max_it,tol, True)



