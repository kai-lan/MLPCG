import os, sys
sys.path.insert(1, 'lib')
import matplotlib.pyplot as plt
from lib.read_data import *
import numpy as np
import torch

path = os.path.dirname(os.path.realpath(__file__))

def vis_flags(frame, shape):
    file_flags = os.path.join(example_folder, f"flags_{frame}.bin")
    flags = read_flags(file_flags)
    flags = flags.reshape(shape)
    plt.imshow(flags.T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/flags_{DIM}d_{N}.png", bbox_inches="tight")
    plt.close()
    return flags

def vis_flags_binary(frame, shape, num_imgs):
    file_flags = os.path.join(example_folder, f"flags_{frame}.bin")
    flags = read_flags(file_flags)
    flags = flags.reshape(shape)
    flags = convert_to_binary_images(flags, num_imgs)
    plt.imshow(flags[0].T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/flags_{DIM}d_{N}_air.png", bbox_inches="tight")
    plt.close()
    plt.imshow(flags[1].T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/flags_{DIM}d_{N}_fluid.png", bbox_inches="tight")
    plt.close()
    plt.imshow(flags[2].T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/flags_{DIM}d_{N}_solid.png", bbox_inches="tight")
    plt.close()
    return flags

def vis_div_v(frame, masked=False):
    file_rhs = os.path.join(example_folder, f"div_v_star_{frame}.bin")
    rhs = load_vector(file_rhs)
    if masked: rhs = abs(rhs) > 1e-16
    plt.imshow(rhs.reshape((N,)*DIM).T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/div_v_star_{DIM}d_{N}.png", bbox_inches="tight")
    plt.close()
    return rhs

def vis_pressure(frame, masked=False):
    file_sol = os.path.join(example_folder, f"pressure_{frame}.bin")
    sol = load_vector(file_sol)
    if masked: sol = abs(sol) > 1e-16
    plt.imshow(sol.reshape((N,)*DIM).T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/pressure_{DIM}d_{N}.png", bbox_inches="tight")
    plt.close()
    return sol

def vis_A(frame):
    file_A = os.path.join(example_folder, f"A_{frame}.bin")
    A = readA_sparse(file_A)

    plt.spy(A)
    plt.savefig(f"A_{DIM}d_{N}.png")
    plt.close()

def plot_loss(data_path, suffix):
    # loss_train = np.load(data_path + f"/training_loss_{suffix}.npy")
    # loss_test = np.load(data_path + f"/validation_loss_{suffix}.npy")
    checkpt = torch.load(data_path+f"/checkpt_{suffix}.tar")
    loss_train = checkpt['training_loss']
    loss_test = checkpt['validation_loss']
    print(loss_train)
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(loss_train, label="train")
    axes[1].plot(loss_test, label="validation")
    plt.xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[0].set_title("Training")
    axes[1].set_title("Validation")
    plt.savefig(f"{path}/loss.png", bbox_inches="tight")

def visualize_frame_by_frame(num_frames, shape):
    import time
    time.sleep(1)
    for i in range(1, num_frames+1):
        vis_flags(i, shape)
        time.sleep(0.1)

if __name__ == '__main__':
    N = 1024
    DIM = 2

    example_folder = os.path.join(DATA_PATH, f"waterflow_rotating_cube_N{N}_200")
    # example_folder = '../TGSL/tgsl/projects/incompressible_flow/build_3D/dambreak_bunny_N64_200_3D'

    frame = 111
    if len(sys.argv) > 1:
        frame = int(sys.argv[1])
    flags = vis_flags(frame, (N, N))
    # flags = vis_flags_binary(frame, (N*3, N), 3)
    print(flags.min(), flags.max())

    # plot_loss(f"{OUT_PATH}/output_{DIM}D_64", "dambreak_M100_ritz100_rhs200_res_binary")
    # rhs = vis_div_v(frame, masked=False)
    # sol = vis_pressure(frame, masked=False)
    # visualize_frame_by_frame(200, (N, N))
    # res = np.load(example_folder + "/b_res_60.npy")
    # weight = vis_weight(frame)
    # plt.imshow(res.reshape((N,)*DIM, order='F'), origin='lower', cmap='jet')
    # plt.colorbar()
    # plt.savefig("res.png")
    # fluids = np.where(flags == 2)
    # print(fluids)

