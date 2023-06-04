import os, sys
sys.path.insert(1, 'lib')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
import torch
from lib.read_data import *
from tqdm import tqdm
from multiprocessing import Process
from numba import njit, prange
import time
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_grad_enabled(False)

DIM = 2
N = 1024
# scale = 2


matrices = range(1, 2)
scene = 'dambreak'
if DIM == 2:
    data_folder = f"{DATA_PATH}/{scene}_N{N}_200"
else:
    data_folder = f"{DATA_PATH}/{scene}_N{N}_200_{DIM}D"
# matrices = np.load(f"{data_folder}/train_mat.npy")
num_ritz_vectors = 1600
num_rhs = 800

def createTrainingData(N, DIM, ritz_vectors, sample_size, fluid_cells, outdir, suffix=''):
    # small_matmul_size = 100 # Small mat size for temp data
    small_matmul_size = sample_size
    theta = 50 # j < m/2 + theta low frequency spectrum

    for_outside = int(sample_size/small_matmul_size)
    b_rhs_temp = np.zeros([small_matmul_size, len(fluid_cells)])
    cut_idx = int(num_ritz_vectors * 0.6)
    sample_size = small_matmul_size
    coef_matrix = np.zeros([len(ritz_vectors), sample_size])

    for it in range(for_outside):
        coef_matrix[:] = np.random.normal(0, 1, [len(ritz_vectors), sample_size])
        coef_matrix[0:cut_idx] *= 9
        b_rhs_temp[:] = coef_matrix.T @ ritz_vectors
        l_b = small_matmul_size * it
        r_b = small_matmul_size * (it+1)
        for i in range(l_b, r_b):
            b_rhs_temp[i-l_b] = b_rhs_temp[i-l_b]/np.linalg.norm(b_rhs_temp[i-l_b])
            # b = torch.zeros(N**DIM, dtype=torch.float32)
            b = torch.tensor(b_rhs_temp[i-l_b], dtype=torch.float32)
            torch.save(b, os.path.join(outdir, f"b_{i}{suffix}.pt"))

def createRandomVec(N, DIM, sample_size, A, air_cells, outdir, suffix='_rand'):
    for i in range(sample_size):
        b = torch.randn(N**DIM)
        b[air_cells] = 0
        b = A @ b
        b /= b.norm()
        torch.save(b, f"{outdir}/b_{i}{suffix}.pt")

def createPerturbedVecFromRHS(N, DIM, rhs, ritz_vecs, sample_size, fluid_cells, outdir, suffix='_perb'):
    rhs = rhs / np.linalg.norm(rhs)
    b = torch.zeros(N**DIM)
    b[fluid_cells] = torch.tensor(rhs[fluid_cells], dtype=torch.float32)
    b /= b.norm()
    eps = 1
    torch.save(b, f"{outdir}/b_0{suffix}.pt")

    coef_matrix = np.random.normal(0, 1, [len(ritz_vecs), sample_size])
    B = coef_matrix.T @ ritz_vecs

    for i in range(1, sample_size):
        b = torch.zeros(N**DIM)
        b[fluid_cells] = eps * torch.tensor(B[i], dtype=torch.float32) + torch.tensor(rhs[fluid_cells], dtype=torch.float32)
        b /= b.norm()
        torch.save(b, f"{outdir}/b_{i}{suffix}.pt")


def createKrylovVec(N, DIM, x0, sample_size, fluid_cells, A, outdir, suffix='_k'):
    krylov_vecs = torch.zeros(sample_size, len(fluid_cells))

    b = x0[fluid_cells]
    b /= b.norm()
    krylov_vecs[0] = b
    # torch.save(b, f"{outdir}/b_0{suffix}.pt")
    for i in range(1, sample_size):
        b = A @ b
        b /= b.norm()
        krylov_vecs[i] = b
    coeff = torch.rand((sample_size, sample_size))
    bs = coeff @ krylov_vecs
    for i in range(sample_size):
        b = torch.zeros(N**DIM, dtype=torch.float32)
        b[fluid_cells] = torch.tensor(bs[i], dtype=torch.float32)
        b /= b.norm()
        torch.save(b, f"{outdir}/b_{i}{suffix}.pt")

def createFourierVecs(N, DIM, num_modes_per_dir, sample_size, flags, A, outdir, suffix='_fourier'):
    num_modes = num_modes_per_dir**DIM
    fluid_cells = np.where(flags == 2)[0]
    flags = flags.reshape((N,)*DIM)
    boundary = []
    for i in range(N):
        for j in range(N):
            if flags[i][j] == 2:
                if (i-1 >= 0 and flags[i-1][j] == 3 ) \
                or (i+1 < N and flags[i+1][j] == 3) \
                or (j-1 >= 0 and flags[i, j-1] == 3) \
                or (j+1 < N and flags[i, j+1] == 3):
                    boundary.append([i, j])
    boundary = np.array(boundary).T

    # np.where(flags_sp.reshape((N,)*DIM) == 2)
    eigen_modes = np.zeros((num_modes, len(fluid_cells)))
    x = np.linspace(0.5/N, (N-0.5)/N, N)
    if DIM == 2:
        X, Y = np.meshgrid(x, x)
    else:
        X, Y, Z = np.meshgrid(x, x, x)
    c = 5
    # @njit(parallel=True)
    def iter_part(eigen_modes):
        nonlocal A
        # for n in range(num_modes_per_dir):
        #     for m in range(num_modes_per_dir):
                # z = (np.cos((c*n+1)*np.pi*X) * np.cos((c*m+1)*np.pi*Y)).ravel()
                # eigen_modes[n * num_modes_per_dir + m, :] = z[fluid_cells]
        counter = 0
        for nm in range(2*num_modes_per_dir-1):
            for n in range(max(0, nm-(num_modes_per_dir-1)), min(nm+1, num_modes_per_dir)):
                m = nm - n
                z = (np.cos((c*n+1)*np.pi*X) * np.cos((c*m+1)*np.pi*Y)) #.ravel()
                # print(boundary[0].shape, z[boundary[0]][:, boundary[1]].shape)
                # z[boundary[0], :][:, boundary[1]] = 0
                # print(z[boundary[0]][:, boundary[1]].shape)
                z = A @ z.ravel()
                z = z.ravel()
                eigen_modes[counter] = z[fluid_cells]
                counter += 1
    @njit(parallel=True)
    def iter_part_3d(eigen_modes):
        for n in range(num_modes_per_dir):
            for m in range(num_modes_per_dir):
                for l in range(num_modes_per_dir):
                    z = (np.cos((c*n+1)*np.pi*X) * np.cos((c*m+1)*np.pi*Y) * np.cos((c*l+1)*np.pi*Z)).ravel()
                    eigen_modes[(n * num_modes_per_dir + m)*num_modes_per_dir + l, :] = z[fluid_cells]
    if DIM == 2:
        iter_part(eigen_modes)
    else:
        iter_part_3d(eigen_modes)
    # coeff = np.random.normal(0, 1, (sample_size, num_modes))
    # # cut_idx = int(num_modes * 0.6)
    # # coeff[:, :cut_idx] *= 0.9
    # result = coeff @ eigen_modes
    np.save(f"{outdir}/fourier.npy", eigen_modes)
    # for i in range(sample_size):
    #     # b = torch.zeros(N**DIM, dtype=torch.float32)
    #     # b[fluid_cells] = torch.tensor(result[i], dtype=torch.float32)
    #     b = torch.tensor(result[i], dtype=torch.float32)
    #     b /= b.norm()
    #     torch.save(b, f"{outdir}/b_{i}{suffix}.pt")

def worker(indices):
    print('Process id', os.getpid())

    for index in indices:
        out_folder = f"{data_folder}/preprocessed/{index}"
        rhs_np = load_vector(os.path.join(data_folder, f"div_v_star_{index}.bin"))
        sol = load_vector(os.path.join(data_folder, f"pressure_{index}.bin"))
        flags_sp = read_flags(os.path.join(data_folder, f"flags_{index}.bin"))
        fluid_cells = np.where(flags_sp == 2)[0]
        air_cells = np.where(flags_sp == 3)[0]

        fluid_cells_md = np.where(flags_sp.reshape((N,)*DIM) == 2)
        # print(fluid_cells_md.shape)
        np.save(f"{out_folder}/fluid_cells.npy", fluid_cells)
        np.save(f"{out_folder}/fluid_cells_md.npy", fluid_cells_md)

        A_sp = readA_sparse(os.path.join(data_folder, f"A_{index}.bin"), sparse_type='csr')

        # A_sp = torch.sparse_coo_tensor(np.array([A_sp.row, A_sp.col]), A_sp.data, A_sp.shape, dtype=torch.float32)
        A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32)
        torch.save(A, os.path.join(out_folder, f"A.pt"))

        flags = torch.tensor(flags_sp, dtype=torch.float32)
        torch.save(flags, os.path.join(out_folder, f"flags.pt"))


        rhs = torch.tensor(rhs_np, dtype=torch.float32)
        torch.save(rhs, os.path.join(out_folder, f"rhs.pt"))

        sol = torch.tensor(sol, dtype=torch.float32)
        torch.save(sol, os.path.join(out_folder, f"sol.pt"))

        # createKrylovVec(N, DIM, rhs, 100, fluid_cells, A_comp, out_folder)
        # ppc = torch.tensor(ppc, dtype=torch.float32)
        # ppc[ppc < 3] = 0
        # torch.save(ppc, os.path.join(out_folder, f"ppc.pt"))

        # levelset = torch.tensor(levelset, dtype=torch.float32)
        # torch.save(levelset, os.path.join(out_folder, f"levelset.pt"))

        # createFourierVecs(N, DIM, 5, 400, flags_sp, A_sp, out_folder)
        ritz_vec = np.load(f"{out_folder}/ritz_{num_ritz_vectors}_no_ortho.npy")
        createTrainingData(N, DIM, ritz_vec, num_rhs, fluid_cells, out_folder, suffix='_no_ortho')
        # createRandomVec(N, DIM, num_rhs, A, air_cells, out_folder)
        # createPerturbedVecFromRHS(N, DIM, rhs_np, ritz_vec, num_rhs, fluid_cells, out_folder)


if __name__ == '__main__':
    t0 = time.time()
    # total_work = range(start, end)
    total_work = matrices
    num_threads = 1
    chunks = np.array_split(total_work, num_threads)
    thread_list = []
    for thr in range(num_threads):
        thread = Process(target=worker, args=(chunks[thr],))
        thread_list.append(thread)
        thread_list[thr].start()
    for thread in thread_list:
        thread.join()
    print('Total time', time.time() - t0)
