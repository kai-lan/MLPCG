# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 00:34:19 2021

@author: osman
"""

import numpy as np
from tensorflow import keras
import os
import tensorflow as tf
import struct
import scipy.sparse as sparse

import pressure_laplacian as pl

        


def QTQ(Q):
    dot_products = np.zeros([Q.shape[1],Q.shape[1]])
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            #This is just to look at the normality
            dot_products[i][j] = np.dot(Q[:,i],Q[:,j])
    return dot_products

def QTQ_norm(Q):
    s = 0.0
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            s = s + abs(np.dot(Q[:,i],Q[:,j]))
    return s

def orthonormality_check(Q):
    s = 0.0
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            if i == j:
                s = s + abs(np.dot(Q[i],Q[j])-1)  
            else:
                s = s + abs(np.dot(Q[i],Q[j]))  
    return s

def diagonality_check(A):
    return np.linalg.norm(A-np.identity(A.shape[0]))
    
    
def normalize_preconditioner(Q):
    for i in range(Q.shape[0]):
        Q[i] = Q[i]/np.linalg.norm(Q[i])
        

def AQ(A,mult_precond):
    M = np.zeros(A.shape)
    for i in range(A.shape[0]):
        M[:,i] = mult_precond(A[i])
    return M


def off_diagonal_sum(A):
    return sum(sum(A)) - sum(A.diagonal())


def get_vec(file_rhs,normalize = False, d_type='double'):
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        r1 = np.delete(r0, [0])
        if normalize:
            return r1/np.linalg.norm(r1)
        else: 
            return r1            

def get_frame_from_source(n,data_folder_name,normalize = True, d_type='double'):
    file_rhs = data_folder_name + "div_v_star_"+str(n)+".bin"
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        #print(r0[0])
        r0 = np.delete(r0, [0])
        if normalize:
            return r0/np.linalg.norm(r0)
        return r0            
    else:
        print("No file for n = "+str(n)+" in data folder "+data_folder_name)

    
def get_frame(n,dim=65,pc="windows", type_=""):
    if pc=="windows":
        data_folder_name = "C:/Users/osman/OneDrive/research_teran/python/data/incompressible_flow_outputs3/output"+str(dim-1)
    elif pc=="mac":
        data_folder_name = "/Users/osmanakar/OneDrive/research_teran/python/data/incompressible_flow_outputs3/output"+str(dim-1)
    
    if type_ == "":    
        data_folder_name = data_folder_name + "/"
    else:
        data_folder_name = data_folder_name + "_"+type_+"/"
    
    file_rhs = data_folder_name + "div_v_star_"+str(n)+".bin"
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype='double')
        r1 = np.delete(r0, [0])
        norm_r = np.linalg.norm(r1)
        return r1/norm_r
    else:
        print("No file for n = "+str(n))

"""
template <typename T>
void Serialize(const std::vector<T>& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  size_t v_size = v.size();
  out.write((char*)&v_size, sizeof(size_t));
  out.write((char*)&v[0], v_size * sizeof(T));
  out.close();
}
"""

def readA(dim,filenameA):
    dim2 = dim*dim;
    mat_A = np.zeros((dim2,dim2));
    with open(filenameA, 'rb') as f:
        length = 8;
        b = f.read(length);
        val = struct.unpack('N', b);
        for j in range(val[0]):
            lenght = 8;
            bj = f.read(lenght);
            ele = struct.unpack('N',bj);
            for k in range(ele[0]):
                len_double = 8;
                bk = f.read(len_double);
                elejk = struct.unpack('d',bk);
                mat_A[j][k] = elejk[0];
    return mat_A

"""
template <typename T, int OptionsBitFlag, typename Index>
void Serialize(SparseMatrix<T, OptionsBitFlag, Index>& m, const std::string& filename) {
  typedef Eigen::Triplet<T, Index> Trip;

  std::vector<Trip> res;

  fstream writeFile;
  writeFile.open(filename, ios::binary | ios::out);

  if (writeFile.is_open()) {
    Index rows, cols, nnzs, outS, innS;
    rows = m.rows();
    cols = m.cols();
    nnzs = m.nonZeros();
    outS = m.outerSize();
    innS = m.innerSize();

    writeFile.write((const char*)&(rows), sizeof(Index));
    writeFile.write((const char*)&(cols), sizeof(Index));
    writeFile.write((const char*)&(nnzs), sizeof(Index));
    writeFile.write((const char*)&(outS), sizeof(Index));
    writeFile.write((const char*)&(innS), sizeof(Index));

    writeFile.write((const char*)(m.valuePtr()), sizeof(T) * m.nonZeros());
    writeFile.write((const char*)(m.outerIndexPtr()), sizeof(Index) * m.outerSize());
    writeFile.write((const char*)(m.innerIndexPtr()), sizeof(Index) * m.nonZeros());

    writeFile.close();
  }
}
"""
def readA_sparse(dim, filenameA, dtype = 'd'):                                                                                                                                                              
    dim2 = dim**3
    cols = []
    outerIdxPtr = []
    rows = []
    if dtype == 'd':
        len_data = 8        
    elif dtype == 'f':
        len_data = 4     
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        #print("nnz = ",nnz)
        #print("num_rows = ", num_rows)
        #print("num_cols = ",num_cols)
        #print("outS = ", outS)
        #print("innS = ", innS)
        data = [0.0] * nnz
        outerIdxPtr = [0]*outS
        cols = [0]*nnz
        rows = [0]*nnz
        for i in range(nnz):
            #length = 8
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
     
    #print(num_rows, num_cols, nnz, outS, innS) 
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        #print(ii," ; ",outerIdxPtr[ii])
        rows[outerIdxPtr[ii]:outerIdxPtr[ii+1]] = [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
     
     
    #print(nnz,len(rows),len(cols),len(data))
    #print(outerIdxPtr[num_rows-1])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2])


def readA_sparse_from_bin(dim, filenameA, dtype = 'd'):
    dim2 = dim**3
    cols = []
    outerIdxPtr = []
    rows = []
    if dtype == 'd':
        len_data = 8
    elif dtype == 'f':
        len_data = 4    
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        #print("nnz = ",nnz)
        #print("num_rows = ", num_rows)
        #print("num_cols = ",num_cols)
        #print("outS = ", outS)
        #print("innS = ", innS)
        data = [0.0] * nnz
        outerIdxPtr = [0]*outS
        cols = [0]*nnz
        rows = [0]*nnz
        for i in range(nnz):
            #length = 8
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
     
    print(num_rows, dim2, nnz, outS) 
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        #print(ii," ; ",outerIdxPtr[ii])
        rows[outerIdxPtr[ii]:outerIdxPtr[ii+1]] = [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
     
     
    #print(nnz,len(rows),len(cols),len(data))
    #print(outerIdxPtr[num_rows-1])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2])
     





def ReadEigenSpMat(dim, filenameA):
    cols = []
    outerIdxPtr = []
    data = []
    rows = []
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        print("nnz = ",nnz)
        print("num_rows = ", num_rows)
        print("num_cols = ",num_cols)
        print("outS = ", outS)
        print("innS = ", innS)
        
        for i in range(nnz):
            length = 8
            b = f.read(length)
            data = data + [struct.unpack('d', b)[0]]
            
        for i in range(outS):
            length = 4
            b = f.read(length)
            outerIdxPtr = outerIdxPtr + [struct.unpack('i', b)[0]]
            
        for i in range(nnz):
            length = 4
            b = f.read(length)
            cols = cols + [struct.unpack('i', b)[0]]
            
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        rows = rows + [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2])

def ReadEigenSpMat(dim, filenameA):
    cols = []
    outerIdxPtr = []
    data = []
    rows = []
    #reading the bit files
    with open(filenameA, 'rb') as f:
        length = 4;
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length);
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length);
        nnz = struct.unpack('i', b)[0]
        b = f.read(length);
        outS = struct.unpack('i', b)[0]
        b = f.read(length);
        innS = struct.unpack('i', b)[0]
        print("nnz = ",nnz)
        print("num_rows = ", num_rows)
        print("num_cols = ",num_cols)
        print("outS = ", outS)
        print("innS = ", innS)
        for i in range(nnz):
            length = 8 
            b = f.read(length)
            data = data + [struct.unpack('d', b)[0]]
                
        for i in range(outS):
            length = 4 
            b = f.read(length)
            outerIdxPtr = outerIdxPtr + [struct.unpack('i', b)[0]]
                
        for i in range(nnz):
            length = 4 
            b = f.read(length)
            cols = cols + [struct.unpack('i', b)[0]]
                
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        rows = rows + [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
    return sparse.csr_matrix((data, (rows, cols)),[dim,dim])

def make_ML_entry(b):
    b_ML = b.copy()
    b_ML = b_ML/np.linalg.norm(b)
    return b_ML.reshape(1,65,65,1)

def check_Q(Q, n=64):
    pres_lap = pl.pressure_laplacian(n)
    return np.matmul(Q,Q.transpose()), np.matmul(np.matmul(Q,pres_lap.A),Q.transpose())


def extend_array(res_arr, mult_coeff):
    res_arr0 = [res_arr[0]]
    for i in range(1,len(res_arr)):
        res_arr0 = res_arr0 + [res_arr[i]]*mult_coeff
    return res_arr0



def create_plot_bar_arr(eigvectors_coef, bar_range):
    res_arr_bar_plot = []
    dim2 = len(eigvectors_coef)
    block_size = int(dim2/bar_range)
    eigvectors_coef2 = eigvectors_coef**2
    for i in range(bar_range):
        res_arr_bar_plot = res_arr_bar_plot + [np.sqrt(sum(eigvectors_coef2[i*block_size:(i+1)*block_size]))]
    return res_arr_bar_plot



def load_model_from_source(model_file_source):
    json_file = open(model_file_source + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ml_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    ml_model.load_weights(model_file_source + "model.h5")
    print("Loaded model from disk")
    return ml_model

def load_model_from_machine(epoch_num, project_folder_general,project_folder_subname, dim=64 , threeD=False,  best_models=False):
    dim2 = dim**2      
    
    if threeD:
        legion_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/MLCG_3D_N"+str(dim)+"/"
        model_name = project_folder_subname+"/saved_models/MLCG_3D_N"+str(dim)+"_json_E"+str(epoch_num)+"/ " +project_folder_general+"saved_models/"
    
    if dim==64:
        legion_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1/"
        model_name = project_folder_subname+"/saved_models/preconditioner_first_iter_MLV1_json_E"+str(epoch_num)+"/ " +project_folder_general+"saved_models/"
    elif dim==128:
        legion_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N127/"
        model_name = project_folder_subname+"/saved_models/preconditioner_first_iter_MLV1_N127_json_E"+str(epoch_num)+"/ " +project_folder_general+"/saved_models/" 
    elif dim==256:
        legion_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N255/"
        model_name = project_folder_subname+"/saved_models/preconditioner_first_iter_MLV1_N255_json_E"+str(epoch_num)+"/ " +project_folder_general+"/saved_models/"
    elif dim==512:
        legion_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N511/"
        model_name = project_folder_subname+"/saved_models/preconditioner_first_iter_MLV1_N511_json_E"+str(epoch_num)+"/ " +project_folder_general+"/saved_models/" 
    else:
        print("No such folder in legion machine")
    
    

    os_return_val = os.system("scp -r "+ legion_folder_name+model_name)
    if os_return_val != 0:
        print("File not found in the legion machine")
        return
    
    if threeD:
        model_name = project_folder_general + "saved_models/MLCG_3D_N"+str(dim)+"_json_E"+str(epoch_num)+'/'

    if dim==64:
        if best_models:
            model_name = project_folder_general + 'saved_models_best/preconditioner_first_iter_MLV1_json_E'+str(epoch_num)+'/'
        else:
            model_name = project_folder_general + 'saved_models/preconditioner_first_iter_MLV1_json_E'+str(epoch_num)+'/'
    elif dim==128:
        if best_models:
            model_name = project_folder_general + 'saved_models_best/preconditioner_first_iter_MLV1_N127_json_E'+str(epoch_num)+'/'
        else:
            model_name = project_folder_general + 'saved_models/preconditioner_first_iter_MLV1_N127_json_E'+str(epoch_num)+'/'
    elif dim==256:
        if best_models:
            model_name = project_folder_general + 'saved_models_best/preconditioner_first_iter_MLV1_N255_json_E'+str(epoch_num)+'/'
        else:
            model_name = project_folder_general + 'saved_models/preconditioner_first_iter_MLV1_N255_json_E'+str(epoch_num)+'/'
    elif dim==512:
        if best_models:
            model_name = project_folder_general + 'saved_models_best/preconditioner_first_iter_MLV1_N511_json_E'+str(epoch_num)+'/'
        else:
            model_name = project_folder_general + 'saved_models/preconditioner_first_iter_MLV1_N511_json_E'+str(epoch_num)+'/'
       

    json_file = open(model_name + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_PFI_V1 = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_PFI_V1.load_weights(model_name + "model.h5")
    print("Loaded model from disk")
    return model_PFI_V1

def load_model_from_machine_V2(epoch_num, project_folder_general, project_name, project_folder_subname, machine="legion"):
    if machine == "legion": 
        machine_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/"+project_name+"/"+ project_folder_subname+"/saved_models/"+project_name+ "_json_E" +str(epoch_num)+"/"
    if machine == "hyde01":
        machine_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/"+project_name+"/"+ project_folder_subname+"/saved_models/"+project_name+ "_json_E" +str(epoch_num)+"/"

        
    model_folder_name = project_folder_general+"saved_models/"
    os_return_val = os.system("scp -r "+ machine_folder_name+" "+model_folder_name)
    if os_return_val != 0:
        print("File not found in the legion machine")
        return  
    
    model_name = model_folder_name+project_name+"_json_E"+str(epoch_num)+"/"
    json_file = open(model_name + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_PFI_V1 = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_PFI_V1.load_weights(model_name + "model.h5")
    print("Loaded model from disk")
    return model_PFI_V1

def load_model_from_machine_V3(epoch_num, model_saving_place, project_name, project_folder_subname, machine="legion"):
    if machine == "legion": 
        machine_folder_name = "osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/"+project_name+"/"+project_folder_subname+"/saved_models/"+project_name+ "_json_E" +str(epoch_num)+"/"
    if machine == "hyde01":
        machine_folder_name = "oak@hyde01.dabh.io:~/projects/ML_preconditioner_project/"+project_name+"/"+project_folder_subname+"/saved_models/"+project_name+ "_json_E" +str(epoch_num)+"/"

    
    #model_folder_name = project_folder_general+"saved_models/"
    model_name = model_saving_place+project_name + "_json_E"+str(epoch_num)+"/"
    os_return_val = os.system("scp -r "+ machine_folder_name+" " + model_name)
    if os_return_val != 0:
        print("File not found in the legion machine")
        return  
    
    #model_name = model_saving_place+project_name+"_json_E"+str(epoch_num)+"/"
    json_file = open(model_name + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_PFI_V1 = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_PFI_V1.load_weights(model_name + "model.h5")
    print("Loaded model from disk")
    return model_PFI_V1

def first_residual_fast_old(b, Q,CG_):
    #Q = Q.reshape([num_vectors,dim2])
    A_tilde = np.matmul(Q,np.matmul(CG_.A, Q.transpose()))
    return Q.transpose()*np.matmul(Q,b)/A_tilde
    #return x_init 
    
def first_residual_fast(b, q, CG_):
    #Q = Q.reshape([num_vectors,dim2])
    A_tilde = np.matmul(Q,np.matmul(CG_.A, Q.transpose()))
    return Q.transpose()*np.matmul(Q,b)/A_tilde
    #return x_init 

def ML_generated_iterative_subspace_solver(b,ml_model,CG_,max_it=100,verbose=True):
    b1 = b.copy()
    dim2 =len(b)
    dim = int(np.sqrt(dim2))
    x_sol = np.zeros(b.shape)
    res_arr = []
    Q = np.zeros([max_it,dim2])
    for i in range(max_it):
        b1_norm = np.linalg.norm(b1)
        b1_normalized = b1/b1_norm
        b_tf = tf.convert_to_tensor(b1_normalized.reshape([1,dim,dim,1]),dtype=tf.float32)
        q = ml_model.predict(b_tf)[0,:,:,:].reshape([dim2]) #first_residual
        Q[i]=q.copy()
        #q = b1.copy()
        #x = first_residual(b1,Q_)
        x = first_residual_fast_old(b1,q,CG_)
        x_sol = x_sol+x
        b1 = b-CG_.multiply_A_sparse(x_sol)
        res_arr = res_arr + [b1_norm]
        #
        if verbose:
            print(i,b1_norm)
    last_norm = np.linalg.norm(b1)
    print("After max_it = ",max_it, " iterations ML-generated solver converged to ",last_norm)
    return x_sol, res_arr

def ML_generated_iterative_subspace_solver_A_conjugate(b,ml_model,CG_,max_it=100,verbose=True, num_previous_bases = 3):
    b1 = b.copy()
    dim2 =len(b)
    dim = int(np.sqrt(dim2))
    x_sol = np.zeros(b.shape)
    res_arr = []
    Q = np.zeros([max_it, dim2])
    AQ = np.zeros([max_it, dim2])
    Q_A_norms = np.zeros(max_it)
    for i in range(max_it):
        b1_norm = np.linalg.norm(b1)
        b1_normalized = b1/b1_norm
        b_tf = tf.convert_to_tensor(b1_normalized.reshape([1,dim,dim,1]),dtype=tf.float32)
        q = ml_model.predict(b_tf)[0,:,:,:].reshape([dim2]) #first_residual
        q_iter = q.copy()
        for j in range(max(0,i-num_previous_bases),i):            
            q_iter = q_iter - Q[j]*np.dot(q,AQ[j])/Q_A_norms[j]
        Q[i] = q_iter.copy()
        AQ[i] = np.matmul(CG_.A,Q[i])
        Q_A_norms[i] = np.dot(Q[i], AQ[i])
        #q = b1.copy()
        #x = first_residual(b1,Q_)
        x = first_residual_fast_old(b1,q_iter,CG_)
        x_sol = x_sol+x
        b1 = b-np.matmul(CG_.A,x_sol)
        res_arr = res_arr + [b1_norm]
        #
        if verbose:
            print(i,b1_norm)
        
    
    last_norm = np.linalg.norm(b1)
    print("After max_it = ",max_it, " iterations ML-generated solver converged to ",last_norm)
    return x_sol, res_arr #, Q, AQ
    

def ML_generated_iterative_subspace_solver_test(ml_model, bs, CG_, max_it=100, tol=1e-6):
    dim2 = len(bs[0])
    #dim = int(np.sqrt(dim2))
    bad_frames = []
    res_arr_average = np.zeros(max_it)
    for i in range(len(bs)):
        b = bs[i].copy()
        x_sol, res_arr = ML_generated_iterative_subspace_solver(b,ml_model,CG_,max_it,False)
        res_arr_average = res_arr_average + np.array(res_arr)
        if res_arr[-1]>tol:
            print(i,res_arr[-1])
            bad_frames = bad_frames+[i]
    
    res_arr_average = res_arr_average/len(bs)
    
    return res_arr_average, bad_frames
    
    
    
    
    
    
    
    
    
    
    
    



