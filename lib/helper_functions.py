# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 00:34:19 2021

@author: osman

Modified by Kai.
"""

import numpy as np
import os
import struct
import scipy.sparse as sparse

import pressure_laplacian as pl

def load_vector(data_folder_name):
    if(os.path.exists(data_folder_name)):
        r0 = np.fromfile(data_folder_name, dtype='float64')
        r1 = np.delete(r0, [0])
        return r1
    else:
        print("No file for exist named "+data_folder_name)

def get_vec(file_rhs,normalize = False, d_type='double'):
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        r1 = np.delete(r0, [0])
        if normalize:
            return r1/np.linalg.norm(r1)
        else:
            return r1

def get_frame_from_source(n,data_folder_name,normalize = True, d_type='double'):
    file_rhs = data_folder_name
    file_rhs = os.path.join(file_rhs, f"div_v_star_{n}.bin")
    if(os.path.exists(file_rhs)):
        r0 = np.fromfile(file_rhs, dtype=d_type)
        r0 = np.delete(r0, [0])
        if normalize:
            return r0/np.linalg.norm(r0)
        return r0
    else:
        print("No file for n = "+str(n)+" in data folder "+data_folder_name)


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
def readA_sparse(dim, filenameA, DIM=2, dtype='f'):
    '''
    dim: grid points in each dimenstion
    DIM: 2D or 3D
    dtype: 'd', double (8 bytes); 'f', float (4 bytes)
    '''
    dim2 = dim**DIM
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
        data = [0.0] * nnz
        outerIdxPtr = [0]*outS
        cols = [0]*nnz
        rows = [0]*nnz
        for i in range(nnz):
            b = f.read(len_data)
            data[i] = struct.unpack(dtype, b)[0]
        for i in range(outS): # Index pointer
            length = 4
            b = f.read(length)
            outerIdxPtr[i] = struct.unpack('i', b)[0]
        for i in range(nnz): # Col index
            length = 4
            b = f.read(length)
            cols[i] = struct.unpack('i', b)[0]
    outerIdxPtr = outerIdxPtr + [nnz]
    for ii in range(num_rows):
        rows[outerIdxPtr[ii]:outerIdxPtr[ii+1]] = [ii]*(outerIdxPtr[ii+1] - outerIdxPtr[ii])
    return sparse.csr_matrix((data, (rows, cols)),[dim2,dim2], dtype=dtype)

# def load_model_from_source(model_file_source):
#     json_file = open(model_file_source + 'model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     ml_model = keras.models.model_from_json(loaded_model_json)
#     # load weights into new model
#     ml_model.load_weights(model_file_source + "model.h5")
#     print("Loaded model from disk")
#     return ml_model

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    dirname = "./dataset_mlpcg/test_64_3D" #test_matrices_and_vectors/N64" #
    file = dirname + "/smoke_passing_bunny/div_v_star_8.bin"
    # with open(file, 'rb') as f:
    #     b = f.read(2)
    #     num_rows = struct.unpack('H', b)[0]
    #     b = f.read(2)
    #     num_cols = struct.unpack('H', b)[0]
    #     print(num_rows, num_cols)
    # A = readA_sparse(64, , 2, 'f')

    b = load_vector(file)
    print(b.shape)
    print(b.max(), b.min())
    # b = np.load(dirname + "/b_6.npy")
    # with open ("b_2D.txt", 'w') as f:
        # sys.stdout = f
        # A.maxprint = np.inf
        # b.maxprint = np.inf
        # for x in b:
            # print(x)

