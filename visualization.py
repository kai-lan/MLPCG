import os, sys
sys.path.insert(1, 'lib')
import matplotlib.pyplot as plt
from lib.read_data import *
import numpy as np
import torch

path = os.path.dirname(os.path.realpath(__file__))

def vis_weight(frame):
    file_flags = os.path.join(example_folder, f"flags_{frame}.bin")
    weight = compute_weight(file_flags, N, DIM)
    plt.imshow(weight.reshape((N,)*DIM, order='F'), origin='lower')
    plt.colorbar()
    plt.savefig(f"{path}/weight_{DIM}d_{N}.png", bbox_inches="tight")
    plt.close()
    return weight

def vis_flags(frame):
    file_flags = os.path.join(example_folder, f"flags_{frame}.bin")
    flags = read_flags(file_flags)
    flags = flags.reshape((N,)*DIM)
    # flags = np.flipud(flags)
    plt.imshow(flags.T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/flags_{DIM}d_{N}.png", bbox_inches="tight")
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

def vis_levelset(frame, masked=False):
    file = os.path.join(example_folder, f"levelset_{frame}.bin")
    levelset = load_vector(file)
    # if masked:
    #     levelset = levelset == 0
    plt.imshow(levelset.reshape((N,)*DIM).T, origin='lower', cmap='jet', vmin=-0.1, vmax=0.2)
    plt.colorbar()
    plt.savefig(f"{path}/levelset_{DIM}d_{N}.png", bbox_inches="tight")
    plt.close()
    # print(np.sum(levelset))
    return levelset

def vis_ppc(frame):
    cells = read_ppc(os.path.join(example_folder, f"active_cells_{frame}.bin"), os.path.join(example_folder, f"ppc_{frame}.bin"), N, DIM)
    plt.imshow(cells.reshape((N,)*DIM).T, origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig(f"{path}/ppc_{DIM}d_{N}.png", bbox_inches="tight")
    plt.close()
    # print(cells.reshape((N,)*DIM).T)
def vis_A(frame, masked=False):
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

def visualize_frame_by_frame(num_frames):
    import time
    time.sleep(1)
    for i in range(1, num_frames+1):
        vis_flags(i)
        time.sleep(0.1)

if __name__ == '__main__':
    N = 64
    DIM = 2
    example_folder = os.path.join(DATA_PATH, f"dambreak_N{N}_200")


    frame=54
    # vis_flags(frame)
    # levelset = vis_levelset(frame, True)
    # print(levelset)
    vis_ppc(frame)
    # plot_loss(f"{OUT_PATH}/output_{DIM}D_64", "dambreak_M100_ritz100_rhs200_res_binary")
    # rhs = vis_div_v(frame, masked=False)
    # sol = vis_pressure(frame, masked=False)
    # visualize_frame_by_frame(200)
    # res = np.load(example_folder + "/b_res_60.npy")
    # weight = vis_weight(frame)
    # plt.imshow(res.reshape((N,)*DIM, order='F'), origin='lower', cmap='jet')
    # plt.colorbar()
    # plt.savefig("res.png")
    # fluids = np.where(flags == 2)
    # print(fluids)

