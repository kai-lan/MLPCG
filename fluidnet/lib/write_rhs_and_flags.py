'''
File: write_rhs_and_flags.py
File Created: Tuesday, 10th January 2023 1:08:03 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
Last Modified: Wednesday, 11th January 2023 12:27:14 am
--------------
'''
import struct
import torch
from velocity_divergence import velocityDivergence
from GLOBAL_VARS import *

# Write out div v data
# Only support 2D for now
def writeRHSAndFlags(input_file, output_files={}):
    if len(output_files) == 0: return
    with open(input_file, 'rb') as f:
        fhead = struct.unpack('i' * 5, f.read(4 * 5))
        nx = fhead[1]
        ny = fhead[2]
        nz = fhead[3]
        is3D = (fhead[4] == 1)
        assert not is3D, "3D not supported!"
        numel = nx * ny * nz

        # read moves cursor to the end of previous read
        # we don't need to move the cursor using seek
        ld_array = struct.unpack('f' * 3 * numel, f.read(4 * 3 * numel)) # 4 bytes for float32, Ux, Uy
        Ux = torch.FloatTensor(ld_array[:numel])
        Uy = torch.FloatTensor(ld_array[numel:(numel * 2)])
        # p = torch.FloatTensor(ld_array[2 * numel:(numel * 3)])
        if (is3D):
            Uz = torch.FloatTensor(struct.unpack('f' * numel, f.read(4 * numel)))
        flags = torch.IntTensor(struct.unpack('i' * numel, f.read(4 * numel))).float()
        flags.resize_(1, 1, nz, ny, nx)
        # We ALWAYS deal with 5D tensor to make things easier.
        # All tensor are always nbatch x nchan x nz x ny x nx.
        Ux.resize_(1, 1, nz, ny, nx)
        Uy.resize_(1, 1, nz, ny, nx)
        if (is3D):
            Uz.resize_(1, 1, nz, ny, nx)
            U = torch.cat((Ux, Uy, Uz), 1).contiguous()
        else:
            U = torch.cat((Ux, Uy), 1).contiguous()
        div_U = velocityDivergence(U, flags)
        if 'div_U' in output_files.keys():
            torch.save(div_U, output_files['div_U'])
        if 'flags' in output_files.keys():
            torch.save(flags, output_files['flags'])

if __name__ == '__main__':
    # Create torch vectors saves as *.pt in data directory
    for i in range(NUM_IMAGES_TR_2D__):
        for j in range(NUM_FRAMES_TR_2D__):
            file_gt = os.path.join(DATA_TR_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}.bin")
            file_div = os.path.join(DATA_TR_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}_divergent.bin")
            assert os.path.exists(file_gt) and os.path.exists(file_div), f"{file_gt}, {file_div}"
            file_div_U = os.path.join(DATA_TR_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}_div.pt")
            file_flags = os.path.join(DATA_TR_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}_flags.pt")
            files_out = {}
            if not os.path.exists(file_div_U): files_out['div_U'] = file_div_U
            if not os.path.exists(file_flags): files_out['flags'] = file_flags
            writeRHSAndFlags(file_div, files_out)
    for i in range(NUM_IMAGES_TE_2D__):
        for j in range(NUM_FRAMES_TE_2D__):
            file_gt = os.path.join(DATA_TE_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}.bin")
            file_div = os.path.join(DATA_TE_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}_divergent.bin")
            assert os.path.exists(file_gt) and os.path.exists(file_div), f"{file_gt}, {file_div}"
            file_div_U = os.path.join(DATA_TE_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}_div.pt")
            file_flags = os.path.join(DATA_TE_2D_PATH__, f"{i:06}", f"{j*FRAME_INCREMENT__:06}_flags.pt")
            files_out = {}
            if not os.path.exists(file_div_U): files_out['div_U'] = file_div_U
            if not os.path.exists(file_flags): files_out['flags'] = file_flags
            writeRHSAndFlags(file_div, files_out)
    exit()
    # from discrete_laplacian import lap_with_bc, image_to_list
    file_div = "../data/datasets/output_current_model_sphere/tr/000256/000048_div.pt"
    file_flags = "../data/datasets/output_current_model_sphere/tr/000256/000048_flags.pt"
    div_U = torch.load(file_div)
    flags = torch.load(file_flags)
    print(div_U.shape, div_U.dtype)
    print(flags.shape, flags.dtype)
