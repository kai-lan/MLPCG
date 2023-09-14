import os, sys
sys.path.insert(1, 'lib')
import numpy as np
from torch.utils.data import DataLoader
from sm_model import *
from loss_functions import *
from lib.GLOBAL_VARS import *
from lib.dataset import *
from lib.read_data import *
torch.set_grad_enabled(False) # disable autograd globally

def recover_training_and_validation_loss(outdir, model_name, epoches, train_bcs, valid_bcs, loss_fn):
    train_losses, valid_losses = [], []
    model = SmallSMModelDn3D(n=3, num_imgs=3)
    model = model.to(cuda)
    shape = None
    fluid_cells = None
    def transform(x):
        nonlocal fluid_cells, shape
        b = torch.zeros(np.prod(shape), dtype=torch.float32, device=cuda)
        b[fluid_cells] = x
        b = b.reshape(shape)
        return b
    train_set = MyDataset(None, range(num_rhs), transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=0)
    for epo in epoches:
        checkpt = torch.load(os.path.join(outdir, f"{model_name}_{epo}.tar"))
        model_params = checkpt['model_state_dict']
        model.load_state_dict(model_params)

        train_tot_loss = 0.0
        train_num_mat = 0
        for count, (bc, sha, matrices) in enumerate(train_bcs, 1):
            train_num_mat += len(matrices)
            shape = (1,)+sha
            inpdir = f"{DATA_PATH}/{bc}_200_{DIM}D/preprocessed"
            num_matrices = len(matrices)
            for j_mat, j in enumerate(matrices, 1):
                print(f"Epoch: {epo}/{len(epoches)}")
                print(bc, f'{count}/{len(train_bcs)}')
                print('Matrix', j, f'{j_mat}/{num_matrices}')

                train_set.data_folder = os.path.join(f"{inpdir}/{j}")

                A = torch.load(f"{train_set.data_folder}/A.pt")
                image = torch.load(f"{train_set.data_folder}/flags_binary_3.pt").view((3,)+sha)

                fluid_cells = torch.load(f"{train_set.data_folder}/fluid_cells.pt", map_location='cuda')
                for data in train_loader:
                    x_pred = model(image, data) # input: (bs, 1, dim, dim)
                    train_tot_loss += loss_fn(x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells], data[:, 0].flatten(1)[:, fluid_cells], A)
        train_losses.append(train_tot_loss.item() / train_num_mat)

        valid_tot_loss = 0.0
        valid_num_mat = 0
        for count, (bc, sha, matrices) in enumerate(valid_bcs, 1):
            shape = (1,)+sha
            valid_num_mat += len(matrices)
            scene_path = f"{DATA_PATH}/{bc}_200_{DIM}D"
            num_matrices = len(matrices)
            for j_mat, j in enumerate(matrices, 1):
                print(f"Epoch: {epo}/{len(epoches)}")
                print(bc, f'{count}/{len(train_bcs)}')
                print('Matrix', j, f'{j_mat}/{num_matrices}')

                A_sp = readA_sparse(os.path.join(scene_path, f"A_{j}.bin")).astype(np.float64)
                rhs_sp = load_vector(os.path.join(scene_path, f"div_v_star_{j}.bin")).astype(np.float64)
                flags_sp = read_flags(os.path.join(scene_path, f"flags_{j}.bin"))
                fluid_cells = np.argwhere(flags_sp == FLUID).ravel()

                if len(rhs_sp) == np.prod(shape):
                    A_comp = compressedMat(A_sp, flags_sp)
                    rhs_comp = compressedVec(rhs_sp, flags_sp)
                else:
                    A_comp = A_sp
                    rhs_comp = rhs_sp

                flags_sp = convert_to_binary_images(flags_sp, 3)
                A = torch.sparse_csc_tensor(A_comp.indptr, A_comp.indices, A_comp.data, A_comp.shape, dtype=torch.float32, device=cuda)
                rhs = torch.tensor(rhs_comp, dtype=torch.float32, device=cuda)
                rhs = transform(rhs).unsqueeze(0)
                image = torch.tensor(flags_sp, dtype=torch.float32, device=cuda).view(3, *shape[1:])
                fluid_cells = torch.from_numpy(fluid_cells).to(cuda)

                x_pred = model(image, rhs) # input: (bs, 1, dim, dim)
                valid_tot_loss += loss_fn(x_pred.squeeze(dim=1).flatten(1)[:, fluid_cells], rhs[:, 0].flatten(1)[:, fluid_cells], A)
        valid_losses.append(valid_tot_loss.item() / valid_num_mat)

    return train_losses, valid_losses


if __name__ == '__main__':

    DIM = 3
    N = 256
    training_bcs = [
        (f'dambreak_N{N}',                  (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        # (f'dambreak_hill_N{N//2}_N{N}',     (N,)+(N//2,)*(DIM-1),   np.linspace(1, 200, 10, dtype=int)),
        # (f'dambreak_dragons_N{N//2}_N{N}',  (N,)+(N//2,)*(DIM-1),    [1, 6, 10, 15, 21, 35, 44, 58, 81, 101, 162, 188]),
        # (f'ball_cube_N{N}',                 (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        # (f'ball_bowl_N{N}',                 (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        # (f'standing_dipping_block_N{N}',    (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        # (f'standing_rotating_blade_N{N}',   (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        # (f'waterflow_pool_N{N}',            (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)),
        # (f'waterflow_panels_N{N}',          (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:]),
        # (f'waterflow_rotating_cube_N{N}',   (N,)*DIM,               np.linspace(1, 200, 10, dtype=int)[1:])
    ]
    validation_bcs = [
        (f'dambreak_pillars_N{N//2}_N{N}',  (N,)+(N//2,)*(DIM-1),       np.linspace(1, 200, 5, dtype=int)),
        (f'dambreak_bunny_N{N//2}_N{N}',    (N,)+(N//2,)*(DIM-1),       np.linspace(1, 200, 5, dtype=int)),
        (f'waterflow_ball_N{N}',            (N,)*DIM,               np.linspace(1, 200, 5, dtype=int))
    ]

    cuda = torch.device('cuda:0')
    outdir = f"output/output_{DIM}D_{N}"
    epoches = [30, 35]
    num_rhs = 32
    loss_fn = residual_loss
    training_loss, validation_loss = recover_training_and_validation_loss(outdir,
                                          'checkpt_mixedBCs_M97_ritz1600_rhs800_imgs3_lr0.0001_from128',
                                          epoches,
                                          training_bcs,
                                          validation_bcs,
                                          residual_loss)
    print(training_loss, validation_loss)