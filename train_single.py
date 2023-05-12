import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sm_model import *
from model import *


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.append('lib')
    from lib.read_data import *
    from lib.dataset import *
    from cg_tests import *
    from train import train_, saveData, loadData
    import time, timeit
    import warnings
    warnings.filterwarnings("ignore") # UserWarning: Sparse CSR tensor support is in beta state
    torch.set_default_dtype(torch.float32)
    # Make training reproducible
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    N = 64
    DIM = 2
    dim2 = N**DIM
    lr = 0.01
    epoch_num = 100
    cuda = torch.device("cuda") # Use CUDA for training

    image_type = 'flags'
    frame = 100
    # num_ritz = 200
    num_rhs = 100
    b_size = 20
    # levels = 20
    # data_path = f"{DATA_PATH}/full_domain/N{N}"
    data_path = f"{DATA_PATH}/dambreak_N{N}_200/preprocessed/{frame}"

    outdir = os.path.join(OUT_PATH, f"output_single_{DIM}D_{N}")
    os.makedirs(outdir, exist_ok=True)
    suffix = f"frame_{frame}_{image_type}"

    A = torch.load(f"{data_path}/A.pt").to_sparse_csr()
    image = torch.load(f"{data_path}/flags.pt").reshape(1, N, N)
    model = SmallSMModelD3()
    model.move_to(cuda)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    resume = False
    if resume:
        start_ep, model_params, optim_params, training_loss, validation_loss, time_history, grad_history, update_history = loadData(outdir, suffix)
        model.load_state_dict(model_params)
        optimizer.load_state_dict(optim_params)

    else:
        start_ep = 0
        training_loss, validation_loss, time_history, grad_history, update_history = [], [], [], [], []

    loss_fn = model.residual_loss


    train_size = round(0.8 * num_rhs)
    perm = np.random.permutation(num_rhs)
    def transform(x):
        x = x.reshape((1, N, N))
        return x

    train_set = MyDataset(data_path, perm[:train_size], transform, suffix='') # 'k' for Krylov vectors: b, Ab, A^2 b, A^3 b
    valid_set = MyDataset(data_path, perm[train_size:], transform, suffix='')

    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=b_size, shuffle=False)


    for_train = True
    for_test = True
    for _ in range(1, 8):
        if for_train:
            model.train()
            start = time.time()
            training_loss_, validation_loss_, time_history_, grad_history_, update_history_ = train_(image, A, epoch_num, train_loader, valid_loader, model, optimizer, loss_fn)
            training_loss.extend(training_loss_)
            validation_loss.extend(validation_loss_)
            time_history.extend(time_history_)
            grad_history.extend(grad_history_)
            update_history.extend(update_history_)
            saveData(model, optimizer, start_ep+epoch_num, None, outdir, suffix, training_loss, validation_loss, time_history, grad_history, update_history)
            end = time.time()
            plt.clf()
            fig, axes = plt.subplots(2)
            axes[0].plot(training_loss, label='training', c='blue')
            axes[1].plot(validation_loss, label='validation', c='orange')
            axes[0].legend()
            axes[1].legend()
            plt.savefig(f"train_loss_{N}.png")
            plt.clf()
            plt.plot(grad_history)
            plt.savefig(f"train_grad_{N}.png")
            plt.clf()
            plt.plot(update_history)
            plt.savefig(f"train_update_{N}.png")
        if for_test:
            checkpt = torch.load(f"{outdir}/checkpt_{suffix}.tar")

            model.load_state_dict(checkpt['model_state_dict'])
            model.eval()
            rhs = torch.tensor(torch.load(f"{data_path}/rhs.pt"))
            # image = torch.tensor(torch.load(f"{data_path}/{image_type}.pt")).reshape(1, N, N)
            image = image.to(cuda)
            # A = torch.load(f"{data_path}/A.pt")
            A = A.to(cuda)
            rhs = rhs.to(cuda)

            def predict(r):
                with torch.no_grad():
                    r = nn.functional.normalize(r, dim=0)
                    # b = torch.stack([r, image]).view(1, 2, N, N)
                    # x = model(image, r.reshape(1, 1, N, N)).flatten()
                    x = model.eval_forward(image, r.reshape(1, 1, N, N)).flatten()
                return x

            x_fluidnet_res, res_fluidnet_res = dcdm(rhs, A, torch.zeros_like(rhs), predict, max_it=100, tol=1e-4, verbose=True)
            print("Fluidnet", res_fluidnet_res[-1])

            A_sp = readA_sparse(f"{data_path}/../../A_{frame}.bin").astype(np.float32)
            rhs_sp = load_vector(f"{data_path}/../../div_v_star_{frame}.bin").astype(np.float32)
            t0 = timeit.default_timer()
            x, res_history = CG(rhs_sp, A_sp, np.zeros_like(rhs_sp), max_it=1000, tol=1e-4, verbose=False)
            print("CG took", timeit.default_timer()-t0, 's after', len(res_history), 'iterations', res_history[-1])

            with open(f"train_info_N{N}_{suffix}.txt", 'a') as f:
                f.write(f"{_*epoch_num}, {len(res_fluidnet_res)-1}, {len(res_history)}\n")
            plt.clf()
            plt.plot(res_history, label='CG')
            plt.plot(res_fluidnet_res, label='MLPCG')
            plt.legend()
            plt.savefig(f"test_loss_{N}.png")
        if for_train:
            print("Training time", end-start)