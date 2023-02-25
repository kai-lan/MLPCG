'''
File: discrete_laplacian.py
File Created: Wednesday, 4th January 2023 12:42:01 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
Part of the code was modified from 2022 Fall MAT 228A HW.
--------------
'''
import numpy as np
from scipy import sparse
###############################################################################
#
# form the (scaled by dx*dx) 1D Laplacian for Dirichlet boundary conditions on a
#  node-centered grid
#
#   input:  n -- number of grid points (no bdy pts)
#
#   output: L1 -- n x n sparse matrix for discrete Laplacian
#
def lap1d(n, dtype=np.float32):
    v  = np.ones(n, dtype=dtype)
    L1 = sparse.spdiags([v,-2*v,v],[-1,0,1],n,n)
    return L1

###############################################################################
#
#   form the (scaled by dx*dx) matrix for the 2D Laplacian for Dirichlet boundary
#   conditions on a rectangular node-centered nx by ny grid
#
#   input:  nx -- number of grid points in x-direction (no bdy pts)
#           ny -- number of grid points in y-direction
#
#   output: L2 -- (nx*ny) x (nx*ny) sparse matrix for discrete Laplacian
#   L2 = kron(Dxx, I) + kron(I, Dyy)
def lap2d(nx,ny, dtype=np.float32):
    Lx=lap1d(nx, dtype=dtype)
    Ix = sparse.eye(nx, dtype=dtype)

    Ly=lap1d(ny, dtype=dtype)
    Iy=sparse.eye(ny, dtype=dtype)

    L2 = sparse.kron(Iy,Lx) + sparse.kron(Ly,Ix)

    return L2
###############################################################################
#
#   form the (scaled by dx*dx*dx) matrix for the 3D Laplacian for Dirichlet boundary
#   conditions on a rectangular node-centered nx by ny grid
#
#   input:  nx -- number of grid points in x-direction (no bdy pts)
#           ny -- number of grid points in y-direction
#
#   output: L3 -- (nx*ny*nz) x (nx*ny*nz) sparse matrix for discrete Laplacian
#   L3 = kron(Dxx, I, I) + kron(I, Dyy, I) + kron(I, I, Dzz)
def lap3d(nx,ny,nz, dtype=np.float32):
    Lx=lap1d(nx, dtype=dtype)
    Ix = sparse.eye(nx, dtype=dtype)

    Ly=lap1d(ny, dtype=dtype)
    Iy=sparse.eye(ny, dtype=dtype)

    Lz=lap1d(nz, dtype=dtype)
    Iz=sparse.eye(nz, dtype=dtype)

    L3 = sparse.kron(sparse.kron(Lx, Iy),Iz) + sparse.kron(sparse.kron(Ix, Ly),Iz) + sparse.kron(sparse.kron(Ix, Iy),Lz)
    return L3

def flatten_inds(inds, n, include_bd=True):
    inds = np.asarray(inds)
    dim = inds.shape[1]
    if include_bd:
        if dim == 2: return inds[:, 0] * n + inds[:, 1]
        return inds[:, 0] * n**2 + inds[:, 1] * n + inds[:, 2]
    # remove points on the boundary
    rows = ((inds > 0) & (inds < n-1)).prod(axis=1, dtype=bool)
    inds = inds[rows, :]
    if dim == 2:
        if dim == 2:
            return (inds[:, 0] - 1) * (n - 2) + (inds[:, 1] - 1)
        return (inds[:, 0] - 1) * (n - 2)**2 + (inds[:, 1] - 1) * (n - 2) + (inds[:, 2] - 1)

def neighbors(ind, n):
    nb = []
    dim = len(ind)
    if ind[0] - 1 >= 0: nb.append([ind[0]-1, *ind[1:]])
    if ind[0] + 1 < n: nb.append([ind[0]+1, *ind[1:]])
    if ind[-1] - 1 >= 0: nb.append([*ind[:-1], ind[-1]-1])
    if ind[-1] + 1 < n: nb.append([*ind[:-1], ind[-1]+1])
    if dim == 3:
        if ind[1] - 1 >= 0: nb.append([ind[0], ind[1]-1, ind[-1]])
        if ind[1] + 1 < n: nb.append([ind[0], ind[1]+1, ind[-1]])
    return nb

def lap_with_bc(n, dim, solid=[], air=[], bd=[], spd=True, include_bd=True, dtype=np.float32):
    """Generate pressure Laplacian matrix with specified BC
    Args:
        n (int): Number of grids in each dimension.
        dim (int): 2D or 3D
        solid (list, optional): list of indices for solid cells, ie, Neumann. Defaults to [].
        air (list, optional): list of indices for air cells, ie, Dirichlet. Defaults to [].
        spd: negate it to make it SPD.
    Returns:
        scipy csr matrix: pressure Laplacian matrix
    """
    m = n if include_bd else n-2
    if dim == 2:
        A = lap2d(m, m, dtype=dtype).tolil()
    else:
        A = lap3d(m, m, m, dtype=dtype).tolil()
    if len(bd) > 0:
        bd_neighbor_cells = []
        for ind in bd: bd_neighbor_cells.extend(neighbors(ind, n))
        bd_neighbors_inds = flatten_inds(bd_neighbor_cells, n, include_bd)
        for i in bd_neighbors_inds: A[i, i] += 1
        if include_bd:
            bd_inds = flatten_inds(bd, n)
            A[bd_inds] = 0
            A[:, bd_inds] = 0
    if len(solid) > 0:
        solid_neighbor_cells = []
        for ind in solid: solid_neighbor_cells.extend(neighbors(ind, n))
        solid_neighbors_inds = flatten_inds(solid_neighbor_cells, n)
        for i in solid_neighbors_inds: A[i, i] += 1
        solid_inds = flatten_inds(solid, n)
        A[solid_inds] = 0
        A[:, solid_inds] = 0
    if len(air) > 0:
        air_inds = flatten_inds(air, n)
        A[air_inds] = 0
        A[:, air_inds] = 0
    if spd: A *= -1
    return A.tocsr()

# image: For 2- or 3-dimensional numpy array
# Mark solid cells (2)
# TODO air cells
def image_to_list(image):
    solid = np.array(np.where(image == 2)).T
    return solid

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys, os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    from write_data import writeA_sparse
    from read_data import readA_sparse
    N = 64
    DIM = 2
    BC = 'empty'
    include_bd = False
    prefix = '_' if include_bd else ''
    n = N if include_bd else N+2
    bd = []
    if DIM == 2:
        for i in range(n):
            bd.extend([(i, 0), (i, n-1), (0, i), (n-1, i)])
    else:
        for i in range(n):
            for j in range(n):
                bd.extend([(i, j, 0), (i, j, n-1), (0, i, j), (n-1, i, j), (i, 0, j), (i, n-1, j)])
    bd = list(set(bd))

    A = lap_with_bc(n, DIM, bd=bd, include_bd=include_bd, dtype=np.float32)
    writeA_sparse(A, os.path.join(dir_path, "..", f"data_dcdm/{prefix}train_{DIM}D_{N}/A_{BC}.bin"), 'f')

    # B = readA_sparse(n, os.path.join(dir_path, f"../dataset_mlpcg/train_{n}_{DIM}D/A_{BC}.bin"), DIM, 'f')
    # B.maxprint = np.inf
    # with open ('matA_test.txt', 'w') as f:
    #     sys.stdout = f
    #     print(B)
    # plt.spy(A, markersize=2, marker='o')
    # plt.savefig("laplacian_sparsity_2d.png")