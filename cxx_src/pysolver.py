#!/usr/bin/env python
import sys, argparse
# sys.path.append('build')
from build import pyamgcl_ext
import numpy   as np
import scipy.sparse as sp
from scipy.io import mmread, mmwrite
from time import time
# from make_poisson import *

from scipy.sparse.linalg import LinearOperator

class solver(pyamgcl_ext.solver):
    """
    Iterative solver with preconditioning
    """
    def __init__(self, P, prm={}):
        self.P = P
        pyamgcl_ext.solver.__init__(self, self.P, prm)

    def __repr__(self):
        return self.P.__repr__()

    def __call__(self, *args):
        """
        Solves the system for the given system matrix and the right-hand side.

        In case single argument is given, it is considered to be the right-hand
        side. The matrix given at the construction is used for solution.

        In case two arguments are given, the first one should be a new system
        matrix, and the second is the right-hand side. In this case the
        preconditioner passed on construction of the solver is still used. This
        may be of use for solution of non-steady-state PDEs, where the
        discretized system matrix slightly changes on each time step, but the
        preconditioner built for one of previous time steps is still able to
        approximate the system matrix.  This saves time needed for rebuilding
        the preconditioner.

        Parameters
        ----------
        A : the new system matrix (optional)
        rhs : the right-hand side
        """
        if len(args) == 1:
            return pyamgcl_ext.solver.__call__(self, args[0])
        elif len(args) == 2:
            Acsr = args[0].tocsr()
            return pyamgcl_ext.solver.__call__(self, Acsr.indptr, Acsr.indices, Acsr.data, args[1])
        else:
            raise "Wrong number of arguments"

class amgcl(pyamgcl_ext.amgcl):
    """
    Algebraic multigrid hierarchy to be used as a preconditioner
    """
    def __init__(self, A, prm={}):
        """
        Creates algebraic multigrid hierarchy to be used as preconditioner.

        Parameters
        ----------
        A     The system matrix in scipy.sparse format
        prm   Dictionary with amgcl parameters
        """
        Acsr = A.tocsr()
        self.shape = A.shape

        pyamgcl_ext.amgcl.__init__(self, Acsr.indptr, Acsr.indices, Acsr.data, prm)

class timeit:
    profile = {}
    def __init__(self, desc):
        self.desc = desc
        self.tic  = time()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        toc = time()
        timeit.profile[self.desc] = timeit.profile.get(self.desc, 0.0) + (toc - self.tic)

    @staticmethod
    def report():
        print('\n---------------------------------')
        total = sum(timeit.profile.values())
        for k,v in sorted(timeit.profile.items()):
            print('{0:>22}: {1:>8.3f}s ({2:>5.2f}%)'.format(k, v, 100 * v / total))
        print('---------------------------------')
        print('{0:>22}: {1:>8.3f}s'.format('Total', total))

#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(sys.argv[0])

parser.add_argument('-A,--matrix', dest='A', help='System matrix in MatrixMarket format')
parser.add_argument('-f,--rhs',    dest='f', help='RHS in MatrixMarket format')
parser.add_argument('-n,--size',   dest='n', type=int, default=64, help='The size of the Poisson problem to solve when no system matrix is given')
parser.add_argument('-o,--out',    dest='x', help='Output file name')
parser.add_argument('-p,--precond', dest='p', help='preconditioner parameters: key1=val1 key2=val2', nargs='+', default=[])
parser.add_argument('-s,--solver',  dest='s', help='solver parameters: key1=val1 key2=val2', nargs='+', default=[])

args = parser.parse_args(sys.argv[1:])

#----------------------------------------------------------------------------
# if args.A:
#     with timeit('Read problem'):
#         A = mmread(args.A)
#         f = mmread(args.f).flatten() if args.f else np.ones(A.shape[0])
# else:
#     with timeit('Generate problem'):
#         A,f = make_poisson_3d(args.n)

A = sp.csr_matrix(np.random.rand(1000, 1000))
f = np.random.rand(1000)

# Parse parameters
p_prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.p)}
s_prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.s)}
print(p_prm)
print(s_prm)
# Create solver/preconditioner pair
with timeit('Setup solver'):
    S = solver(amgcl(A, p_prm), s_prm)
print(S)

# Solve the system for the RHS
with timeit('Solve the problem'):
    x = S(f)

error = np.linalg.norm(f - A * x) / np.linalg.norm(f)
print("{0.iters}: {0.error:.6e} / {1:.6e}".format(S, error))
print(S.iters)
# Save the solution
# if args.x:
#     with timeit('Save the result'):
#         mmwrite(args.x, x.reshape((-1,1)))

# timeit.report()
