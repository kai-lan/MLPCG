import os, sys
sys.path.insert(1, 'lib')
import torch
from lib.read_data import *
from tqdm import tqdm
from multiprocessing import Process
import time

DIM = 2
N = 64
data_folder = os.path.join(DATA_PATH, f"dambreak_{DIM}D_{N}")

start = 1
end = 1001
num_rhs = 200


def worker(indices):
    print('Process id', os.getpid())

    for index in indices:
        file_rhs = os.path.join(data_folder, f"div_v_star_{index}.bin")
        file_flags = os.path.join(data_folder, f"flags_{index}.bin")
        file_A = os.path.join(data_folder, f"A_{index}.bin")
        rhs = load_vector(file_rhs)
        flags = read_flags(file_flags)

        # Stored the compressed matrix and vectors (with empty cells removed)
        out_folder = os.path.join(data_folder, "preprocessed", str(index))
        A = readA_sparse(file_A).tocoo()
        # Creating tensors from a list of np arrays is slow
        # CSC and CSR do not support stacking for batching, but COO does
        A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data, A.shape, dtype=torch.float32)
        torch.save(A, os.path.join(out_folder, f"A.pt"))

        flags = torch.tensor(flags, dtype=torch.float32)
        torch.save(flags, os.path.join(out_folder, f"flags.pt"))

        rhs = torch.tensor(rhs, dtype=torch.float32)
        torch.save(rhs, os.path.join(out_folder, f"rhs.pt"))

        for i in range(num_rhs):
            # b = np.load(f"{out_folder}/b_{i}.npy")
            b = sparse.load_npz(f"{out_folder}/b_{i}.npz").toarray().ravel()
            b = torch.tensor(b, dtype=torch.float32)
            x = torch.stack([b, flags])
            torch.save(x, os.path.join(out_folder, f"x_{i}.pt"))

if __name__ == '__main__':
    t0 = time.time()
    total_work = range(start, end)
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
