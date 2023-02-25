import os
import torch
from lib.read_data import *
from tqdm import tqdm

def preprocess(data_folder, f_start, f_end, dtype):
    os.makedirs(os.path.join(data_folder, "preprocessed"), exist_ok=True)
    for index in tqdm(range(f_start, f_end)):
        file_rhs = os.path.join(data_folder, f"div_v_star_{index}.bin")
        file_flags = os.path.join(data_folder, f"flags_{index}.bin")
        file_A = os.path.join(data_folder, f"A_{index}.bin")
        rhs = torch.tensor(load_vector(file_rhs), dtype=dtype)
        flags = torch.tensor(read_flags(file_flags), dtype=dtype)
        # weight = torch.tensor(compute_weight(file_flags, N, DIM), dtype=dtype)
        x = torch.stack([rhs, flags])
        A = readA_sparse(file_A).tocoo()
        # Creating tensors from a list of np arrays is slow
        # CSC and CSR do not support stacking for batching, but COO does
        A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data, A.shape, dtype=dtype)
        # torch.save(weight, os.path.join(data_folder, "preprocessed", f"w_{index}.pt"))
        if dtype == torch.float32: suffix = 'f'
        if dtype == torch.float64: suffix = 'd'
        torch.save(x, os.path.join(data_folder, "preprocessed", f"x_{index}_{suffix}.pt"))
        torch.save(A, os.path.join(data_folder, "preprocessed", f"A_{index}_{suffix}.pt"))

if __name__ == '__main__':
    DIM = 2
    N = 64
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "data_fluidnet", f"dambreak_{DIM}D_{N}")
    start = 1
    end = 1001
    preprocess(folder, start, end, torch.float32)
