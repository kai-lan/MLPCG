import os, sys
sys.path.insert(1, 'lib')
import torch
from lib.read_data import *
from tqdm import tqdm


DIM = 2
N = 64
data_folder = os.path.join(DATA_PATH, f"largedambreak_{DIM}D_{N}")
start = 1
end = 1001
for index in tqdm(range(start, end)):
    file_rhs = os.path.join(data_folder, f"div_v_star_{index}.bin")
    file_flags = os.path.join(data_folder, f"flags_{index}.bin")
    file_A = os.path.join(data_folder, f"A_{index}.bin")
    rhs = torch.tensor(load_vector(file_rhs), dtype=torch.float32)
    flags = torch.tensor(read_flags(file_flags), dtype=torch.float32)
    # weight = torch.tensor(compute_weight(file_flags, N, DIM), dtype=dtype)
    # x = torch.stack([rhs, flags])
    A = readA_sparse(file_A).tocoo()
    # Creating tensors from a list of np arrays is slow
    # CSC and CSR do not support stacking for batching, but COO does
    A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data, A.shape, dtype=torch.float32)
    # torch.save(weight, os.path.join(data_folder, "preprocessed", f"w_{index}.pt"))
    out_folder = os.path.join(data_folder, "preprocessed", str(index))
    for i in range(1000):
        b = np.load(f"{out_folder}/b_{i}.npy")
        b = torch.tensor(b, dtype=torch.float32)
        x = torch.stack([b, flags])
        torch.save(x, os.path.join(out_folder, f"x_{i}.pt"))
    torch.save(flags, os.path.join(out_folder, f"flags.pt"))
    torch.save(rhs, os.path.join(out_folder, f"rhs.pt"))
    torch.save(A, os.path.join(out_folder, f"A.pt"))
