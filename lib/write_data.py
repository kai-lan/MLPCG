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
        #xx = np_array.copy();
        #xx = np.concatenate([[size_t_size], xx])
        s = struct.pack('d'*len(np_array), *np_array)
        out_file.write(s)
