"""

"""
import struct
import numpy as np

"""
filename_orig = "GitHub/extra/incompressible_flow_outputs/output64_MLoutput/"
filename = filename_orig+str(1)+'.bin'
x1 = np.fromfile(filename, dtype='double')
size_t_size = x1[0];
"""

# Get a binary file from RHS:
def read_bin_file_into_nparray(filename,normalize = "yes"):
    b1 = np.fromfile(filename, dtype='double')
    np_array = np.delete(b1, [0])
    if normalize is "yes":
        norm_arr = np.linalg.norm(np_array)
        np_array = np_array/norm_arr
    return np_array
    
def write_bin_file_from_nparray(filename,np_array):
    out_file = open(filename,"wb")
    size_arr = [len(np_array)]
    s_size = struct.pack('N'*1,*size_arr)
    out_file.write(s_size)
    #xx = np_array.copy();
    #xx = np.concatenate([[size_t_size], xx])
    s = struct.pack('d'*len(np_array), *np_array)
    out_file.write(s)
    out_file.close()
    




