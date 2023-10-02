import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('lib')
from lib.read_data import *
import struct
import numpy as np
import scipy.sparse as sparse
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.sparse import sparse_dense_matmul as sp_mult

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
    start_time = time.time()
    res_arr = []
    time_arr = []
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
    time_arr = [time.time()-start_time]
    for i in range(max_it):
        r_normalized = r / norm_r
        q = model_predict(r_normalized)
        # q = r_normalized
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
        time_arr = time_arr + [time.time()-start_time]
        if verbose:
            print(i+1, norm_r)
        if norm_r < tol:
            print("DCDM converged in ", i+1, " iterations to residual ", norm_r)
            for i in reversed(range(len(time_arr))):
                time_arr[i] -= time_arr[0]
            return x_sol, res_arr, time_arr

    print("DCDM converged in ", max_it, "(maximum iteration) iterations to residual ",norm_r)
    for i in reversed(range(len(time_arr))):
        time_arr[i] -= time_arr[0]
    return x_sol, res_arr, time_arr


gpus = tf.config.list_physical_devices('GPU')

print(gpus)

N = 256
N2 = N**3
dim = N+2
dim2 = dim**3
frames = range(200, 201)
max_it = 100
tol = 1e-6


scene = f"waterflow_ball_N{N}_200_3D"
# scene = f"smoke_bunny_N{N}_200_3D"
data_dir = f"data/{scene}"


model_dir = f"../dataset_mlpcg/trained_models/model_N{N}_from128_F32"
json_file = open(model_dir + '/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = get_predefined_model(dim, 'from128')
model.load_weights(model_dir + "/model.h5")

def model_predict(r):
    if compressed:
        rr = np.zeros(N2)
        rr[fluid_cells] = r.numpy()
        rr = tf.convert_to_tensor(rr, tf.float32)
    else:
        rr = tf.cast(r, tf.float32)
    r = tf.reshape(rr, [1, N,N,N])
    r = tf.pad(r, [[0,0],[1,1], [1,1], [1,1]])
    x = model(r,training=False)
    # x = r
    x = x[0, 1:-1, 1:-1, 1:-1]
    x = tf.reshape(x, [-1])
    if compressed:
        x = tf.convert_to_tensor(x.numpy()[fluid_cells], tf.float64)
    else:
        x = tf.cast(x, tf.float64)
    return x

output_file = f"output/output_3D_{N}/dcdm_{scene}.txt"
for frame in frames:
    b = load_vector(f"{data_dir}/div_v_star_{frame}.bin")
    A = readA_sparse(f"{data_dir}/A_{frame}.bin",'d', shape=None)
    flags = read_flags(f"{data_dir}/flags_{frame}.bin")
    fluid_cells = np.where(flags==FLUID)[0]
    compressed = len(b) < N2

    b = tf.convert_to_tensor(b, dtype=tf.float64)
    b /= tf.norm(b)
    A = A.tocoo().astype(np.float64)
    A = tf.sparse.SparseTensor(np.array([A.row, A.col]).T, A.data, A.shape)

    x_sol, res_arr, time_arr = dcdm(A, b, tf.zeros_like(b), model_predict, max_it,tol, False)
    t0=time.time()
    x_sol, res_arr, time_arr = dcdm(A, b, tf.zeros_like(b), model_predict, 600,tol, True)
    time_dcdm = time.time() - t0
    print("DCDM took ",time_dcdm, " secs")
    with open(output_file, 'a') as f:
        f.write(f"{frame:>4}, {len(res_arr):>4}, {time_dcdm:>4.3f}\n")
