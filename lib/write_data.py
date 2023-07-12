'''
File: write_data.py
File Created: Thursday, 5th January 2023 1:11:21 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
Last Modified: Thursday, 5th January 2023 1:13:18 am
--------------
'''
import struct

def write_bin_file_from_nparray(filename, np_array):
    with open(filename, "wb") as out_file:
        size_arr = [len(np_array)]
        s_size = struct.pack('N'*1,*size_arr)
        out_file.write(s_size)
        s = struct.pack('d'*len(np_array), *np_array)
        out_file.write(s)

def writeA_sparse(A, filenameA, dtype='f'):
    '''
    dim: grid points in each dimenstion
    DIM: 2D or 3D
    dtype: 'd', double (8 bytes); 'f', float (4 bytes)
    '''
    num_rows, num_cols = A.shape
    nnz = A.nnz
    outS = len(A.indptr)
    innS = outS
    with open(filenameA, 'wb') as f:
        b = struct.pack('i', num_rows)
        f.write(b)
        b = struct.pack('i', num_cols)
        f.write(b)
        b = struct.pack('i', nnz)
        f.write(b)
        b = struct.pack('i', outS)
        f.write(b)
        b = struct.pack('i', innS)
        f.write(b)
        for i in range(nnz):
            b = struct.pack(dtype, A.data[i])
            f.write(b)
        for i in range(outS): # Index pointer
            b = struct.pack('i', A.indptr[i])
            f.write(b)
        for i in range(nnz): # Col index
            b = struct.pack('i', A.indices[i])
            f.write(b)

