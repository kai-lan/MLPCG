import os, sys
sys.path.insert(1, 'lib')
os.environ['OMP_NUM_THREADS'] = '4'
import torch
from lib.read_data import *
from tqdm import tqdm
from multiprocessing import Process
import time
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_grad_enabled(False)

DIM = 2
N = 512
# scale = 2


matrices = range(1, 2)
scene = 'wedge'
if DIM == 2:
    data_folder = f"{DATA_PATH}/{scene}_N{N}_200"
else:
    data_folder = f"{DATA_PATH}/{scene}_N{N}_200_{DIM}D"
# matrices = np.load(f"{data_folder}/train_mat.npy")
num_ritz_vectors = 3200
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
            b = torch.zeros(N**DIM, dtype=torch.float32)
            b[fluid_cells] = torch.tensor(b_rhs_temp[i-l_b], dtype=torch.float32)
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

def worker(indices):
    print('Process id', os.getpid())

    for index in indices:
        out_folder = f"{data_folder}/preprocessed/{index}"
        rhs_np = load_vector(os.path.join(data_folder, f"div_v_star_{index}.bin"))
        sol = load_vector(os.path.join(data_folder, f"pressure_{index}.bin"))
        flags = read_flags(os.path.join(data_folder, f"flags_{index}.bin"))
        fluid_cells = np.where(flags == 2)[0]
        air_cells = np.where(flags == 3)[0]
        # ppc = read_ppc(os.path.join(data_folder, f"active_cells_{index}.bin"), os.path.join(data_folder, f"ppc_{index}.bin"), N, DIM)

        # levelset = load_vector(os.path.join(data_folder, f"levelset_{index}.bin"))

        A = readA_sparse(os.path.join(data_folder, f"A_{index}.bin"), sparse_type='csr')

        # A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data, A.shape, dtype=torch.float32)
        A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, A.shape, dtype=torch.float32)
        torch.save(A, os.path.join(out_folder, f"A.pt"))

        flags = torch.tensor(flags, dtype=torch.float32)
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

        ritz_vec = np.load(f"{out_folder}/ritz_{num_ritz_vectors}.npy")
        createTrainingData(N, DIM, ritz_vec, num_rhs, fluid_cells, out_folder)
        # createRandomVec(N, DIM, num_rhs, A, air_cells, out_folder)
        # createPerturbedVecFromRHS(N, DIM, rhs_np, ritz_vec, num_rhs, fluid_cells, out_folder)


if __name__ == '__main__':
    t0 = time.time()
    # total_work = range(start, end)
    total_work = matrices
    num_threads = 4
    chunks = np.array_split(total_work, num_threads)
    thread_list = []
    for thr in range(num_threads):
        thread = Process(target=worker, args=(chunks[thr],))
        thread_list.append(thread)
        thread_list[thr].start()
    for thread in thread_list:
        thread.join()
    print('Total time', time.time() - t0)
