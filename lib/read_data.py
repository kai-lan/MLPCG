'''
File: read_data.py
File Created: Thursday, 5th January 2023 12:53:43 am

Modified from Osman's code.
--------------
'''
import os
from GLOBAL_VARS import *
import struct
import numpy as np
import scipy.sparse as sparse

SOLID = 1
FLUID = 2
AIR = 3

def read_flags(file, dtype='int32'):
    len_size_t = 8
    if dtype == 'int32': length = 4
    else: length = 8
    with open(file, 'rb') as f:
        N = struct.unpack('N', f.read(len_size_t))[0]  # First 8 bytes stores length of vector
        b = struct.unpack(f'{N}i', f.read(length * N))
    b = np.array(b, dtype=dtype)
    return b

def convert_to_binary_images(image, num_imgs):
    if num_imgs == 1:
        return np.where(image == FLUID, 1, 0).reshape((1,)+image.shape)
    if num_imgs == 2:
        air_img = np.where(image == AIR, 1, 0)
        solid_img = np.where(image <= SOLID, 0, 1)
        return np.stack([air_img, solid_img])
    elif num_imgs == 3:
        air_img = np.where(image == AIR, 1, 0)
        fluid_img = np.where(image == FLUID, 1, 0)
        solid_img = np.where(image <= SOLID, 0, 1)
        return np.stack([air_img, fluid_img, solid_img])

def load_vector(data_folder_name, normalize = False, dtype='double'):
    if(os.path.exists(data_folder_name)):
        r0 = np.fromfile(data_folder_name, dtype=dtype)
        r1 = np.delete(r0, [0])
        if normalize: return r1/np.linalg.norm(r1)
        else:         return r1
    else:
        print("No file for exist named " + data_folder_name)

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
def readA(dim, filenameA, DIM, dtype='d'):
    N = dim**DIM
    mat_A = np.zeros((N,N), dtype=dtype)
    with open(filenameA, 'rb') as f:
        length = 8
        b = f.read(length)
        val = struct.unpack('N', b)
        for j in range(val[0]):
            lenght = 8
            bj = f.read(lenght)
            ele = struct.unpack('N',bj)
            for k in range(ele[0]):
                len_double = 8
                bk = f.read(len_double)
                elejk = struct.unpack('d',bk)
                mat_A[j][k] = elejk[0]
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
def readA_sparse(filenameA, dtype='d', sparse_type='csr'):
    '''
    dim: grid points in each dimenstion
    DIM: 2D or 3D
    dtype: 'd', double (8 bytes); 'f', float (4 bytes)
    '''
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
    if sparse_type.lower() == 'csr':
        return sparse.csr_matrix((data, (rows, cols)),[num_rows, num_cols], dtype=dtype)
    elif sparse_type.lower() == 'coo':
        return sparse.coo_matrix((data, (rows, cols)), [num_rows, num_cols], dtype=dtype)
    else:
        raise Exception("Sparse type only supports coo or csr")

def compressedMat(A, flags):
    return A[A.getnnz(1)>0][:,A.getnnz(0)>0]
def compressedVec(b, flags):
    selection = np.where(flags.ravel() == 2)[0]
    return b[selection]
def expandVec(b, flags):
    flags = flags.ravel()
    selection = np.where(flags == 2)[0]
    v = np.zeros(flags.shape, b.dtype)
    v[selection] = b
    return v

if __name__ == '__main__':
    frame = 1
    N = 64
    DIM = 2
    prefix = ''
    bc = 'dambreak'
    if DIM == 2:
        suffix = ''
    else:
        suffix = '_3D'
    file_A = os.path.join(DATA_PATH, f"{prefix}{bc}_N{N}_200{suffix}", f"A_{frame}.bin")
    file_rhs = os.path.join(DATA_PATH, f"{prefix}{bc}_N{N}_200{suffix}", f"div_v_star_{frame}.bin")
    file_sol = os.path.join(DATA_PATH, f"{prefix}{bc}_N{N}_200{suffix}", f"pressure_{frame}.bin")
    file_flags = os.path.join(DATA_PATH, f"{prefix}{bc}_N{N}_200{suffix}", f"flags_{frame}.bin")
    A = readA_sparse(file_A)
    rhs = load_vector(file_rhs)
    sol = load_vector(file_sol)

    b = load_vector('div_v_star_10.bin')
    print(b.shape)
    # print(A.shape)
    # flags = read_flags(file_flags)
    # flags_binray = convert_to_binary_images(flags)
    # print(flags_binray.shape)
    # air = np.where(flags == 3)[0]
    # fluid = np.where(flags == 2)[0]
    # solid = np.where(flags == 0)[0]


    # print('fluid cells:', len(fluid), 'air cells:', len(air), 'solid cells', len(solid))

    # A = compressedMat(A, flags)
    # rhs = compressedVec(rhs, flags)
    # sol = compressedVec(sol, flags)
    # print(len(rhs))
    print(rhs.shape, A.shape, sol.shape)
    r = rhs - A @ sol
    r_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(rhs)
    print(r_norm, r_norm/b_norm)

