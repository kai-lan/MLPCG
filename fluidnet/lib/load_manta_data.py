import struct
import torch
import numpy as np
# data indexes     |           |
#       (dim 1)    |    2D     |    3D
# ----------------------------------------
#   DATA:
#       pDiv       |    0      |    0
#       UDiv       |    1:3    |    1:4
#       flags      |    3      |    4
#       densityDiv |    4      |    5
#   TARGET:
#       p          |    0      |    0
#       U          |    1:3    |    1:4
#       density    |    3      |    4

#   enum CellType {
#         TypeNone     = 0,
#         TypeFluid    = 1,
#         TypeObstacle = 2,
#         TypeEmpty    = 4,
#         TypeInflow   = 8,
#         TypeOutflow  = 16,
#         TypeOpen     = 32,
#         TypeStick    = 64,

#         TypeReserved = 256,

#    }
def loadMantaFile(fname):
    with open(fname, 'rb') as f:
        fhead = struct.unpack('i' * 5, f.read(4 * 5))
        nx = fhead[1]
        ny = fhead[2]
        nz = fhead[3]
        is3D = (fhead[4] == 1)

        numel = nx * ny * nz

        # read moves cursor to the end of previous read
        # we don't need to move the cursor using seek
        ld_array = struct.unpack('f' * 3 * numel, f.read(4 * 3 * numel))
        Ux = torch.FloatTensor(ld_array[:numel])
        Uy = torch.FloatTensor(ld_array[numel:(numel * 2)])
        p = torch.FloatTensor(ld_array[2 * numel:(numel * 3)])
        if (is3D):
            Uz = torch.FloatTensor(struct.unpack('f' * numel, f.read(4 * numel)))

        flags = torch.IntTensor(struct.unpack('i' * numel, f.read(4 * numel))).float()
        density = torch.FloatTensor(struct.unpack('f' * numel, f.read(4 * numel)))

        # We ALWAYS deal with 5D tensor to make things easier.
        # All tensor are always nbatch x nchan x nz x ny x nx.
        Ux.resize_(1, 1, nz, ny, nx)
        Uy.resize_(1, 1, nz, ny, nx)
        if (is3D):
            Uz.resize_(1, 1, nz, ny, nx)
        p.resize_(1, 1, nz, ny, nx)
        flags.resize_(1, 1, nz, ny, nx)
        density.resize_(1, 1, nz, ny, nx)

        if (is3D):
            U = torch.cat((Ux, Uy, Uz), 1).contiguous()
        else:
            U = torch.cat((Ux, Uy), 1).contiguous()

        return p, U, flags, density, is3D

def loadMantaFileNumpy(fname):
    with open(fname, 'rb') as f:
        fhead = struct.unpack('i' * 5, f.read(4 * 5))
        nx = fhead[1]
        ny = fhead[2]
        nz = fhead[3]
        is3D = (fhead[4] == 1)

        numel = nx * ny * nz

        # read moves cursor to the end of previous read
        # we don't need to move the cursor using seek
        ld_array = struct.unpack('f' * 3 * numel, f.read(4 * 3 * numel))
        Ux = np.array(ld_array[:numel], dtype=np.float32)
        Uy = np.array(ld_array[numel:(numel * 2)], dtype=np.float32)
        p = np.array(ld_array[2 * numel:(numel * 3)], dtype=np.float32)
        if (is3D):
            Uz = np.array(struct.unpack('f' * numel, f.read(4 * numel)))
            U = np.vstack([Ux, Uy, Uz])
        else:
            U = np.vstack([Ux, Uy])
        flags = np.array(struct.unpack('i' * numel, f.read(4 * numel)), dtype=np.int32)
        density = np.array(struct.unpack('f' * numel, f.read(4 * numel)), dtype=np.int32)
        return p, U, flags, density, is3D

if __name__ == '__main__':
    file = "../data/datasets/output_current_model_sphere/tr/000256/000048.bin"
    p, U, flags, density, is3D = loadMantaFile(file)
    print(U)
