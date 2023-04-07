import os, sys
sys.path.insert(1, 'lib')
import torch
from lib.read_data import *
from tqdm import tqdm
from multiprocessing import Process
import time

DIM = 2
N = 64
data_folder = os.path.join(DATA_PATH, f"dambreak_N{N}_200")

matrices = np.load(f"{data_folder}/train_mat.npy")
# start = 10
# end = 11
num_rhs = 200
num_ritz_vectors = 100

def createTrainingData(ritz_vectors, sample_size, fluid_cells, outdir):
    # small_matmul_size = 100 # Small mat size for temp data
    small_matmul_size = sample_size
    theta = 200 # j < m/2 + theta low frequency spectrum

    for_outside = int(sample_size/small_matmul_size)
    b_rhs_temp = np.zeros([small_matmul_size, len(fluid_cells)])
    cut_idx = int(num_ritz_vectors/2) + theta
    sample_size = small_matmul_size
    coef_matrix = np.zeros([len(ritz_vectors), sample_size])

    # print("Creating Dataset ")
    # t0=time.time()
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
            # s = sparse.coo_matrix((b_rhs_temp[i-l_b], (padding, np.zeros_like(padding))), shape=flags.shape+(1,), dtype=np.float32)
            torch.save(b, os.path.join(outdir, f"b_{i}.pt"))

    # print("Creating training Dataset took", time.time() - t0, 's')

def worker(indices):
    print('Process id', os.getpid())

    for index in indices:
        rhs = load_vector(os.path.join(data_folder, f"div_v_star_{index}.bin"))
        flags = read_flags(os.path.join(data_folder, f"flags_{index}.bin"))
        fluid_cells = np.where(flags == 2)[0]

        ppc = read_ppc(os.path.join(data_folder, f"active_cells_{index}.bin"), os.path.join(data_folder, f"ppc_{index}.bin"), N, DIM)

        levelset = load_vector(os.path.join(data_folder, f"levelset_{index}.bin"))

        # Stored the compressed matrix and vectors (with empty cells removed)
        out_folder = os.path.join(data_folder, "preprocessed", str(index))
        A = readA_sparse(os.path.join(data_folder, f"A_{index}.bin"))

        # Creating tensors from a list of np arrays is slow
        # CSC and CSR do not support stacking for batching, but COO does
        # A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data, A.shape, dtype=torch.float32)
        A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, A.shape, dtype=torch.float32)
        torch.save(A, os.path.join(out_folder, f"A.pt"))

        flags = torch.tensor(flags, dtype=torch.float32)
        torch.save(flags, os.path.join(out_folder, f"flags.pt"))

        rhs = torch.tensor(rhs, dtype=torch.float32)
        torch.save(rhs, os.path.join(out_folder, f"rhs.pt"))

        ppc = torch.tensor(ppc, dtype=torch.float32)
        torch.save(ppc, os.path.join(out_folder, f"ppc.pt"))

        levelset = torch.tensor(levelset, dtype=torch.float32)
        torch.save(levelset, os.path.join(out_folder, f"levelset.pt"))

        ritz_vec = np.memmap(f"{out_folder}/ritz_{num_ritz_vectors}.dat", dtype=np.float32, mode='r').reshape(num_ritz_vectors, len(fluid_cells))
        createTrainingData(ritz_vec, num_rhs, fluid_cells, out_folder)

        # for i in range(num_rhs):
        #     # b = sparse.load_npz(f"{out_folder}/b_{i}.npz").toarray().ravel()
        #     b = torch.tensor(b[], dtype=torch.float32)
        #     x = torch.stack([b, flags])
        #     torch.save(x, os.path.join(out_folder, f"x_{i}.pt"))

if __name__ == '__main__':
    t0 = time.time()
    # total_work = range(start, end)
    total_work = matrices
    num_threads = 8
    chunks = np.array_split(total_work, num_threads)
    thread_list = []
    for thr in range(num_threads):
        thread = Process(target=worker, args=(chunks[thr],))
        thread_list.append(thread)
        thread_list[thr].start()
    for thread in thread_list:
        thread.join()
    print('Total time', time.time() - t0)
