'''
File: datasetcreate.py
File Created: Saturday, 25th February 2023 1:27:07 am

Author: Kai Lan (kai.weixian.lan@gmail.com)

Create Ritz vectors (approximated eigenvectors) for training.
https://en.wikipedia.org/wiki/Lanczos_algorithm suggested the reduced number of vectors
should be selected to be approximately 1.5 times the number of accurate eigenvalues desired.
For symmetric matrix, we first reduce the matrix A to tridiagonal (H = Q^T A Q),
then compute eigenvalues (a_0, a_1...) and eigenvectors (u_0, u_1...) for H, then Qu_j is a
Ritz vector for matrix A.

--------------
'''
#OMP_NUM_THREADS=8 python ....py
from GLOBAL_VARS import *
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import scipy
import time

import read_data as hf
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
from tqdm import tqdm

def lanczos_algorithm(A, rhs, num_ritz_vec, ortho_iters=0, cut_off_tol=1e-10):
    return _lanczos_algorithm(A, rhs, num_ritz_vec, ortho_iters=ortho_iters, cut_off_tol=cut_off_tol)
# Given A n x n, return V, T with A = VTV^T
def _lanczos_algorithm(A, init_v, num_ritz_vec, ortho_iters=0, cut_off_tol=1e-10):
    assert A.shape[0] == A.shape[1], "A is not square"
    n = A.shape[0]
    # m = min(num_ritz_vec+10, n)  # Generate 1.5 times number of ritz vectors of the desired
    m = num_ritz_vec
    V = np.zeros((m, n)) # Store orthonormal vectors v0, v1, ... as row vectors (Numpy array is row-major, so accessing row is faster
    alpha = np.zeros(m) # Diagonal of T
    beta = np.zeros(m-1) # Super- and sub-diagonal of T
    V[0] = init_v / np.linalg.norm(init_v) # initialize wit rhs
    w = A @ V[0]
    alpha[0] = w.dot(V[0])
    w = w - alpha[0] * V[0]

    for j in range(1, m):
        beta[j-1] = np.linalg.norm(w)
        if beta[j-1] < cut_off_tol: # cut off if lower than some tolerance
            print("Cut off at", j)
            return V[:j], alpha[:j], beta[:j-1] # Only keep previous entries
        else:
            V[j] = w / beta[j-1]
        w = A @ V[j]
        alpha[j] = w.dot(V[j])
        w = w - alpha[j] * V[j] - beta[j-1] * V[j-1] # later can be used for V[j+1]
        it = min(max(j-1, 0), ortho_iters)
        for k in reversed(range(it)):
            w = w - V[k].dot(w) * V[k]
    return V, alpha, beta

def _test_lanczos_algorithm(A, num_ritz_vec, orthogonal):
    W, diag, subdiag = _lanczos_algorithm(A, num_ritz_vec, orthogonal)
    D = W @ W.T
    D = D - np.identity(D.shape[0])
    print(np.linalg.norm(D))
    E = W @ A @ W.T - sparse.diags([subdiag, diag, subdiag], (-1, 0, 1))
    Es.append(np.linalg.norm(E))
    print(np.linalg.norm(E))

def createRitzVec(A, rhs, num_ritz_vectors):
    # Creating Lanczos Vectors:
    print("Lanczos Iteration is running...")
    start = time.time()
    W, diagonal, sub_diagonal = _lanczos_algorithm(A, rhs, num_ritz_vectors, np.inf)
    print("Lanczos Iteration took", time.time() - start, 's')
    # Calculating eigenvectors of the tridiagonal matrix: only return eigvals > 1e-8
    print("Calculating eigenvectors of the tridiagonal matrix")
    start = time.time()
    ritz_vals, Q = scipy.linalg.eigh_tridiagonal(diagonal, sub_diagonal, select='a')
    # print(ritz_vals.shape, Q.shape)
    print("Calculating eigenvectors took", time.time() - start, 's')
    ritz_vectors = (W.T @ Q[:, :num_ritz_vectors]).T # m x n
    return ritz_vals, ritz_vectors

def createRawData(ritz_vectors, sample_size, flags, outdir):
    if len(ritz_vectors) < sample_size: raise Exception("Given ritz vectors are less than sample size")
    ritz_vectors = ritz_vectors[:sample_size] # Take the lower spectrum
    padding = np.where(flags == 2)[0]
    for i in range(sample_size):
        s = sparse.coo_matrix((ritz_vectors[i], (padding, np.zeros_like(padding))), shape=flags.shape+(1,), dtype=np.float32)
        with open(f"{outdir}/b_{i}.npz", 'wb') as f:
            sparse.save_npz(f, s)

if __name__ == '__main__':
    np.random.seed(2)
    N = 256
    DIM = 2

    dir = f"{DATA_PATH}/dambreak_N{N}_200"
    os.makedirs(dir, exist_ok=True)
    num_ritz_vectors = 800
    # start_frame = 10
    # end_frame = 11
    # perm = np.random.permutation(range(1, 201)) #[:100]
    # np.save(f"{dir}/train_mat.npy", perm)
    perm = range(1, 201)

    for i in tqdm(perm):
        print('Matrix', i)
        out = f"{dir}/preprocessed/{i}"
        os.makedirs(out, exist_ok=True)
        A = hf.readA_sparse(f"{dir}/A_{i}.bin")
        flags = hf.read_flags(f"{dir}/flags_{i}.bin")
        rhs = hf.load_vector(f"{dir}/div_v_star_{i}.bin")
        sol = hf.load_vector(f"{dir}/pressure_{i}.bin")

        A = hf.compressedMat(A, flags)
        rhs = hf.compressedVec(rhs, flags)
        # rhs = np.random.rand(A.shape[0])
        ritz_vals, ritz_vec = createRitzVec(A, rhs, num_ritz_vectors)
        print(ritz_vals[:20])
        np.save(f"{out}/ritz_{num_ritz_vectors}.npy", ritz_vec)

        # _test_lanczos_algorithm(A, num_ritz_vectors, or)
        # createResVec(A, rhs, verbose=True)
