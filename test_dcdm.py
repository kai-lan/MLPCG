import os, sys, time
import struct
import numpy as np
import scipy.sparse as sparse
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.sparse import sparse_dense_matmul as sp_mult

def get_frame_from_source(file_rhs, normalize=True, d_type='double'):
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        r0 = np.delete(r0, [0])
        if normalize:
            return r0/np.linalg.norm(r0)
        return r0
    else:
        print(f"File {file_rhs} not exist")

def readA_sparse(dim, filenameA, dtype = 'd',dim_power = 3, shape=None):
    dim2 = dim**dim_power
    cols = []
    outerIdxPtr = []
    rows = []
    if dtype == 'd':
        len_data = 8
    elif dtype == 'f':
        len_data = 4
    with open(filenameA, 'rb') as f:
        length = 4
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length)
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length)
        nnz = struct.unpack('i', b)[0]
        b = f.read(length)
        outS = struct.unpack('i', b)[0]
        b = f.read(length)
        innS = struct.unpack('i', b)[0]
        data = [0.0] * nnz
        outerIdxPtr = [0]*outS
        cols = [0]*nnz
        rows = [0]*nnz
        for i in range(nnz):
            b = f.read(len_data)
            data[i] = struct.unpack(dtype, b)[0]
        for i in range(outS):
            length = 4
            b = f.read(length)
            outerIdxPtr[i] = struct.unpack('i', b)[0]
        for i in range(nnz):
            length = 4
            b = f.read(length)
            cols[i] = struct.unpack('i', b)[0]
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        rows[outerIdxPtr[ii]:outerIdxPtr[ii+1]] = [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
    if shape is None:
        shape = [dim2,dim2]
    return sparse.csr_matrix((data, (rows, cols)),shape=shape)

def get_predefined_model(N, name_model):
    if name_model == "from64":
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
        return model

    elif name_model == "from128":
        fil_num=16
        input_rhs = keras.Input(shape=(N, N, N, 1))
        first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

        apa = layers.AveragePooling3D((2, 2,2), padding='same')(lb) #7
        apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
        apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
        apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
        apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
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
        upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa)
        upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

        last_layer = layers.Dense(1, activation='linear')(upa)

        model = keras.Model(input_rhs, last_layer)
        return model

def dcdm(A, b, x_init, model_predict, max_it=100, tol=1e-10, verbose=True):
    dim2 =len(b)
    res_arr = []
    p0 = tf.zeros_like(b)
    p1 = tf.zeros_like(b)
    Ap0 = tf.zeros_like(b)
    Ap1 = tf.zeros_like(b)
    alpha0 = 1.0
    alpha1 = 1.0
    r = b - tf.reshape(sp_mult(A, tf.reshape(x_init, (dim2, 1))), [-1])
    norm_r = tf.norm(r).numpy()
    res_arr = [norm_r]
    tol = norm_r*tol
    if verbose:
        print("Initial residual =",norm_r)
    if norm_r < tol:
        print("DCDM converged in 0 iterations to residual ",norm_r)
        return x_init, res_arr

    x_sol = x_init
    for i in range(max_it):
        r_normalized = r / norm_r
        q = model_predict(r_normalized)
        q = q - p1 * tf.tensordot(q, Ap1, 1) / alpha1 - p0 * tf.tensordot(q, Ap0, 1) / alpha0
        Ap0 = Ap1
        Ap1 = tf.reshape(sp_mult(A, tf.reshape(q, (dim2, 1))), [-1])
        p0 = p1
        p1 = q
        alpha0 = alpha1
        alpha1 = tf.tensordot(p1, Ap1, 1)
        beta = tf.tensordot(p1, r, 1)
        x_sol = x_sol + p1 * beta/alpha1
        r = b - tf.reshape(sp_mult(A, tf.reshape(x_sol, (dim2, 1))), [-1])
        norm_r = tf.norm(r).numpy()
        res_arr = res_arr + [norm_r]
        if verbose:
            print(i+1, norm_r)
        if norm_r < tol:
            print("DCDM converged in ", i+1, " iterations to residual ", norm_r)
            return x_sol, res_arr

    print("DCDM converged in ", max_it, "(maximum iteration) iterations to residual ",norm_r)
    return x_sol, res_arr


gpus = tf.config.list_physical_devices('GPU')

print(gpus)

N = 128
N2 = N**3
dim = N+2
dim2 = dim**3
frame = 1
max_it = 100
tol = 1e-6

# data_dir = "../dataset_mlpcg/test_matrices_and_vectors/N128/rotating_fluid"
data_dir = f"data/smoke_bunny_N{N}_200_3D"
b = get_frame_from_source(f"{data_dir}/div_v_star_{frame}.bin", False)
x = get_frame_from_source(f"{data_dir}/pressure_{frame}.bin", False)

bb = np.zeros(dim2)
xx = np.zeros(dim2)
indices = np.zeros(len(b), dtype=int)
for i in range(N):
    ii = i + 1
    for j in range(N):
        jj = j + 1
        for k in range(N):
            kk = k + 1
            bb[(ii*dim+jj)*dim+kk] = b[(i*N+j)*N+k]
            xx[(ii*dim+jj)*dim+kk] = x[(i*N+j)*N+k]
            indices[(i*N+j)*N+k] = (ii*dim+jj)*dim+kk

A = readA_sparse(N, f"{data_dir}/A_{frame}.bin",'d', shape=(dim2, dim2))
new_indptr = np.zeros_like(A.indptr)

new_indptr[indices+1] = A.indptr[1:len(indices)+1]
for i in range(1, len(new_indptr)):
    if new_indptr[i] == 0: new_indptr[i] = new_indptr[i-1]

new_indices = indices[A.indices]
A.indptr = new_indptr
A.indices = new_indices

b = tf.convert_to_tensor(bb, dtype=tf.float64)
x = tf.convert_to_tensor(xx, dtype=tf.float64)
A = A.tocoo().astype(np.float64)
A = tf.sparse.SparseTensor(np.array([A.row, A.col]).T, A.data, A.shape)

b /= tf.norm(b)

model_dir = f"../dataset_mlpcg/trained_models/model_N{N}_from128_F32"
json_file = open(model_dir + '/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = get_predefined_model(dim, 'from128')
model.load_weights(model_dir + "/model.h5")

def model_predict(r):
    r = tf.cast(r, tf.float32)
    r = tf.reshape(r, [1,dim,dim,dim])
    x = model(r,training=False)
    x = tf.reshape(x, [-1])
    x = tf.cast(x, tf.float64)
    return x

t0=time.time()
x_sol, res_arr_ml_generated_cg = dcdm(A, b, tf.zeros_like(b), model_predict, max_it,tol, True)
time_dcdm1 = time.time() - t0
print("DCDM took ",time_dcdm1, " secs")