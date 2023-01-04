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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 4
    A = lap2d(n, n)
    print(type(A))
    plt.spy(A, markersize=2, marker='o')
    plt.savefig("laplacian_sparsity_2d.png")