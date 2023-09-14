import os, sys
sys.path.insert(1, 'lib')
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from lib.read_data import *
from tqdm import tqdm
from multiprocessing import Process
import time
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_grad_enabled(False)

DIM = 3
N = 256
num_imgs = 3
device = torch.device("cpu")

# matrices = np.linspace(1, 200, 10, dtype=int)
matrices = np.linspace(12, 188, 9, dtype=int)[5:]
# matrices = [12, 56, 166]
scenes = [
    # f'dambreak_N{N}',
    # f'dambreak_hill_N{N}',
    # f'dambreak_hill_N{N//2}_N{N}',
    # f'dambreak_dragons_N{N//2}_N{N}',
    # f'two_balls_N{N}',
    # f'ball_cube_N{N}',
    # f'ball_bowl_N{N}',
    # f'standing_dipping_block_N{N}',
    # f'standing_rotating_blade_N{N}',
    # f'standing_scooping_N{N}',
    f'waterflow_pool_N{N}',
    # f'waterflow_panels_N{N}',
    # f'waterflow_rotating_cube_N{N}'
]


num_ritz_vectors = 1600
num_rhs = 800

def createTrainingData(N, DIM, ritz_vectors, sample_size, fluid_cells, outdir):
    # small_matmul_size = 100 # Small mat size for temp data
    small_matmul_size = sample_size

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
            b = torch.tensor(b_rhs_temp[i-l_b], dtype=torch.float32, device=device)
            torch.save(b, os.path.join(outdir, f"b_{i}.pt"))



def worker(indices):
    print('Process id', os.getpid())
    for scene in scenes:
        if DIM == 2:
            data_folder = f"{DATA_PATH}/{scene}_200"
        else:
            data_folder = f"{DATA_PATH}/{scene}_200_{DIM}D"
        for index in indices:
            out_folder = f"{data_folder}/preprocessed/{index}"
            os.makedirs(out_folder, exist_ok=True)
            rhs_np = load_vector(os.path.join(data_folder, f"div_v_star_{index}.bin"))
            flags_sp = read_flags(os.path.join(data_folder, f"flags_{index}.bin"))
            flags_binary_sp = convert_to_binary_images(flags_sp, num_imgs)

            fluid_cells = np.where(flags_sp == 2)[0]
            air_cells = np.where(flags_sp == 3)[0]

            np.save(f"{out_folder}/fluid_cells.npy", fluid_cells)
            torch.save(torch.from_numpy(fluid_cells).to(device), os.path.join(out_folder, f"fluid_cells.pt"))

            A_sp = readA_sparse(os.path.join(data_folder, f"A_{index}.bin"), sparse_type='csr')

            # A_sp = compressedMat(A_sp, flags_sp)


            A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32, device=device).to_sparse_csc()
            torch.save(A, os.path.join(out_folder, f"A.pt"))


            flags_binary = torch.tensor(flags_binary_sp, dtype=torch.float32, device=device)
            torch.save(flags_binary, os.path.join(out_folder, f"flags_binary_{num_imgs}.pt"))

            # rhs_np = compressedVec(rhs_np, flags_sp)

            rhs = torch.tensor(rhs_np, dtype=torch.float32, device=device)
            torch.save(rhs, os.path.join(out_folder, f"rhs.pt"))

            ritz_vec = np.load(f"{out_folder}/ritz_{num_ritz_vectors}.npy")

            createTrainingData(N, DIM, ritz_vec, num_rhs, fluid_cells, out_folder)


if __name__ == '__main__':
    t0 = time.time()
    total_work = matrices
    num_threads = len(total_work)
    chunks = np.array_split(total_work, num_threads)
    thread_list = []
    for thr in range(num_threads):
        thread = Process(target=worker, args=(chunks[thr],))
        thread_list.append(thread)
        thread_list[thr].start()
    for thread in thread_list:
        thread.join()
    print('Total time', time.time() - t0)
