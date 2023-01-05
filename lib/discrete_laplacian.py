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
def lap1d(n):
    v  = np.ones(n)
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
def lap2d(nx,ny):
    Lx=lap1d(nx)
    Ix = sparse.eye(nx)

    Ly=lap1d(ny)
    Iy=sparse.eye(ny)

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
def lap3d(nx,ny,nz):
    Lx=lap1d(nx)
    Ix = sparse.eye(nx)

    Ly=lap1d(ny)
    Iy=sparse.eye(ny)

    Lz=lap1d(nz)
    Iz=sparse.eye(nz)

    L3 = sparse.kron(sparse.kron(Lx, Iy),Iz) + sparse.kron(sparse.kron(Ix, Ly),Iz) + sparse.kron(sparse.kron(Ix, Iy),Lz)
    return L3

def flatten_inds(inds, n):
    inds = np.asarray(inds)
    dim = inds.shape[1]
    if dim == 2: return inds[:, 0] * n + inds[:, 1]
    return inds[:, 0] * n**2 + inds[:, 1] * n + inds[:, 2]
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

def lap_with_bc(n, dim, solid=[], air=[]):
    """Generate pressure Laplacian matrix with specified BC
    Args:
        n (int): Number of grids in each dimension.
        dim (int): 2D or 3D
        solid (list, optional): list of indices for solid cells, ie, Neumann. Defaults to [].
        air (list, optional): list of indices for air cells, ie, Dirichlet. Defaults to [].

    Returns:
        scipy csr matrix: pressure Laplacian matrix
    """
    if dim == 2:
        A = lap2d(n, n).tolil()
    else:
        A = lap3d(n, n, n).tolil()
    if len(solid) > 0:
        solid_inds = flatten_inds(solid, n)
        A[solid_inds] = 0
        A[:, solid_inds] = 0
    if len(air) > 0:
        air_neighbor_cells = []
        for ind in air: air_neighbor_cells.extend(neighbors(ind, n))
        air_neighbors_inds = flatten_inds(air_neighbor_cells, n)
        for i in air_neighbors_inds: A[i, i] += 1
        air_inds = flatten_inds(air, n)
        A[air_inds] = 0
        A[:, air_inds] = 0

    return A.tocsr()
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    n = 64
    air = []
    for i in range(n):
        for j in range(n):
            air.extend([(i, j, 0), (i, j, n-1), (0, i, j), (n-1, i, j), (i, 0, j), (i, n-1, j)])
    air = list(set(air))
    A = lap_with_bc(n, 3, air=air)
    # A.eliminate_zeros()
    A.maxprint = np.inf
    with open ('matA_test.txt', 'w') as f:
        sys.stdout = f
        print(A)
    # plt.spy(A, markersize=2, marker='o')
    # plt.savefig("laplacian_sparsity_2d.png")