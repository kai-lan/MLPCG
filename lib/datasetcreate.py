# Create approximated eigenvectors for DCDM training
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import scipy
import time

import conjugate_gradient as cg
import read_data as hf
import scipy.sparse.linalg as slin


def createRitzVec(DIM, N, A):
    sample_size = 2000
    num_ritz_vectors = 1000 # 256
    small_matmul_size = 200 # Small mat size for temp data
    theta = 50 # j < m/2 + theta low frequency spectrum

    # Load the matrix A
    CG = cg.ConjugateGradientSparse(A)
    rand_vec_x = np.random.normal(0,1, [N**DIM])
    rand_vec = A.dot(rand_vec_x)
    # Creating Lanczos Vectors:
    print("Lanczos Iteration is running...")
    W, diagonal, sub_diagonal = CG.lanczos_iteration_with_normalization_correction(rand_vec, num_ritz_vectors) #this can be loaded directly from c++ output
    #W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_ritz_vectors, 1.0e-12) //Without ortogonalization. This is OK for 2D case.
    print("Lanczos Iteration finished.")

    # Calculating eigenvectors of the tridiagonal matrix: only return eigvals > 1e-8
    print("Calculating eigenvectors of the tridiagonal matrix")
    ritz_vals, Q0 = scipy.linalg.eigh_tridiagonal(diagonal, sub_diagonal, select='v', select_range=(1.0e-8, np.inf))
    print(ritz_vals)
    print(len(ritz_vals))
    ritz_vectors = np.matmul(W.transpose(), Q0).transpose()

    for_outside = int(sample_size/small_matmul_size)
    b_rhs_temp = np.zeros([small_matmul_size, N**DIM])
    cut_idx = int(num_ritz_vectors/2) + theta
    sample_size = small_matmul_size
    coef_matrix = np.zeros([len(ritz_vals), sample_size])

    print("Creating Dataset ")
    t0=time.time()
    for it in range(for_outside):
        coef_matrix[:] = np.random.normal(0,1, [len(ritz_vals), sample_size])
        coef_matrix[0:cut_idx] *= 9
        b_rhs_temp[:] = coef_matrix.T @ ritz_vectors #mat_mult(ritz_vectors.transpose(), coef_matrix).transpose()
        l_b = small_matmul_size*it
        r_b = small_matmul_size*(it+1)

        for i in range(l_b,r_b):
            b_rhs_temp[i-l_b] = b_rhs_temp[i-l_b]/np.linalg.norm(b_rhs_temp[i-l_b])
            with open(f"{outdir}/b_{bc}_{i}.npy", 'wb') as f:
                np.save(f, np.array(b_rhs_temp[i-l_b], dtype=np.float32))
    print("Training Dataset is created.")
    print("Took", time.time()-t0, 's')

def createResVec(A, b, tol=1e-20, max_it=300, verbose=False):
    x_init = np.zeros_like(b)
    count = 0
    norm_b = np.linalg.norm(b)
    r = b - A @ x_init
    np.save(f"{outdir}/b_res_{count}.npy", r.astype(np.float32))
    norm_r = np.linalg.norm(r)
    res_history = [norm_r/norm_b]
    if verbose:
        print(f"Iter {count}, residual {norm_r/norm_b}")
    def callback(x):
        nonlocal count
        count += 1
        r = b - A @ x
        np.save(f"{outdir}/b_res_{count}.npy", r.astype(np.float32))
        norm_r = np.linalg.norm(r)
        res_history.append(norm_r/norm_b)
        if verbose:
            print(f"Iter {count}, residual {norm_r/norm_b}")
    x, info = slin.cg(A, b, x0=x_init, tol=tol, maxiter=max_it, callback=callback)
    return x, res_history
if __name__ == '__main__':
    N = 256
    DIM = 2
    include_bd = False
    prefix = '_' if include_bd else ''
    outdir = dir_path + f"/../data_dcdm/{prefix}train_{DIM}D_{N}"
    os.makedirs(outdir, exist_ok=True)
    bc = "dambreak_800"
    A_file_name = f"{dir_path}/../data_fluidnet/dambreak_{DIM}D_{N}/A_800.bin"
    A = hf.readA_sparse(A_file_name, dtype='d')
    createRitzVec(DIM, N, A)

    rhs = hf.load_vector(os.path.join(dir_path, f"../data_fluidnet/dambreak_{DIM}D_{N}", f"div_v_star_800.bin"))
    # createResVec(A, rhs, verbose=True)