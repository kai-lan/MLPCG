import os, sys
sys.path.insert(1, 'lib')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['NUMBA_NUM_THREADS'] = '4'
import torch
from lib.read_data import *
from tqdm import tqdm
from multiprocessing import Process
import time
import warnings
warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
torch.set_grad_enabled(False)

DIM = 3
N = 128
ortho = True
num_imgs = 3


matrices = np.linspace(1, 200, 10, dtype=int)
# matrices = [200]
scenes = [
    f'dambreak_N{N}',
    f'dambreak_hill_N{N}_N{2*N}',
    f'two_balls_N{N}',
    f'ball_cube_N{N}',
    f'ball_bowl_N{N}',
    f'standing_dipping_block_N{N}',
    f'standing_rotating_blade_N{N}',
    f'waterflow_pool_N{N}',
    f'waterflow_panels_N{N}',
    f'waterflow_rotating_cube_N{N}'
]


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
            b = torch.tensor(b_rhs_temp[i-l_b], dtype=torch.float32).cuda()
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

def worker(indices):
    print('Process id', os.getpid())
    for scene in scenes:
        if DIM == 2:
            data_folder = f"{DATA_PATH}/{scene}_200"
        else:
            data_folder = f"{DATA_PATH}/{scene}_200_{DIM}D"
        for index in indices:
            out_folder = f"{data_folder}/preprocessed/{index}"
            rhs_np = load_vector(os.path.join(data_folder, f"div_v_star_{index}.bin"))
            # sol = load_vector(os.path.join(data_folder, f"pressure_{index}.bin"))
            flags_sp = read_flags(os.path.join(data_folder, f"flags_{index}.bin"))
            flags_binary_sp = convert_to_binary_images(flags_sp, num_imgs)

            fluid_cells = np.where(flags_sp == 2)[0]
            air_cells = np.where(flags_sp == 3)[0]

            # fluid_cells_md = np.where(flags_sp.reshape((N,)*DIM) == 2)

            np.save(f"{out_folder}/fluid_cells.npy", fluid_cells)
            # np.save(f"{out_folder}/fluid_cells_md.npy", fluid_cells_md)

            A_sp = readA_sparse(os.path.join(data_folder, f"A_{index}.bin"), sparse_type='csr')

            # A_sp = torch.sparse_coo_tensor(np.array([A_sp.row, A_sp.col]), A_sp.data, A_sp.shape, dtype=torch.float32)
            A = torch.sparse_csr_tensor(A_sp.indptr, A_sp.indices, A_sp.data, A_sp.shape, dtype=torch.float32)
            torch.save(A, os.path.join(out_folder, f"A.pt"))

            flags = torch.tensor(flags_sp, dtype=torch.float32)
            torch.save(flags, os.path.join(out_folder, f"flags.pt"))
            flags_binary = torch.tensor(flags_binary_sp, dtype=torch.float32)
            torch.save(flags_binary, os.path.join(out_folder, f"flags_binary_{num_imgs}.pt"))

            rhs = torch.tensor(rhs_np, dtype=torch.float32)
            torch.save(rhs, os.path.join(out_folder, f"rhs.pt"))

            suffix = '' if ortho else '_no_ortho'
            ritz_vec = np.load(f"{out_folder}/ritz_{num_ritz_vectors}{suffix}.npy")

            createTrainingData(N, DIM, ritz_vec, num_rhs, fluid_cells, out_folder, suffix=suffix)


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
