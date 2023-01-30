'''
File: read_data.py
File Created: Thursday, 5th January 2023 12:53:43 am

Modified from Osman's code.
--------------
'''
import os
import struct
import numpy as np
import scipy.sparse as sparse

def read_flags(file, dtype='int32'):
    len_size_t = 8
    if dtype == 'int32': length = 4
    else: length = 8
    with open(file, 'rb') as f:
        N = struct.unpack('N', f.read(len_size_t))[0]  # First 8 bytes stores length of vector
        b = struct.unpack(f'{N}i', f.read(length * N))
    b = np.array(b, dtype=dtype)
    return b

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
def readA_sparse(dim, filenameA, DIM=2, dtype='d'):
    '''
    dim: grid points in each dimenstion
    DIM: 2D or 3D
    dtype: 'd', double (8 bytes); 'f', float (4 bytes)
    '''
    N = dim**DIM
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
    return sparse.csr_matrix((data, (rows, cols)),[N,N], dtype=dtype)

if __name__ == '__main__':
    path = os.path.dirname(os.path.relpath(__file__))
    frame = 700
    file_A = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"A_{frame}.bin")
    file_rhs = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"div_v_star_{frame}.bin")
    file_sol = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"pressure_{frame}.bin")
    file_flags = os.path.join(path,  "..", "data_fluidnet", "dambreak_2D_64", f"flags_{frame}.bin")
    A = readA_sparse(64, file_A, DIM=2)
    rhs = load_vector(file_rhs)
    sol = load_vector(file_sol)
    flags = read_flags(file_flags)
    r = rhs - A @ sol
    print(np.linalg.norm(r))

    # file = os.path.join(path,  "..", "data_dcdm", "train_2D_64", f"A_solid.bin")
    # A = readA_sparse(64, file, DIM=2, dtype='f')
    # print(A)
    # with open("matA_test.txt", 'w') as f:
    #     sys.stdout = f
    #     A.maxprint = np.inf
    #     print(A)
