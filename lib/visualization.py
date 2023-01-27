import matplotlib.pyplot as plt
from read_data import read_flags, load_vector
import os

path = os.path.dirname(os.path.relpath(__file__))
N = 64
DIM = 2
example_folder = os.path.join(path,  "..", "data_fluidnet", f"dambreak_{DIM}D_{N}")

def vis_flags(frame):
    file_flags = os.path.join(example_folder, f"flags_{frame}.bin")
    flags = read_flags(file_flags)
    plt.imshow(flags.reshape((N,)*DIM).T, origin='lower')
    plt.colorbar()
    plt.savefig(f"flags_{DIM}d_{N}.png")
    plt.close()
    return flags

def vis_div_v(frame, masked=False):
    file_rhs = os.path.join(example_folder, f"div_v_star_{frame}.bin")
    rhs = load_vector(file_rhs)
    if masked: rhs = abs(rhs) > 1e-16
    plt.imshow(rhs.reshape((N,)*DIM).T, origin='lower')
    plt.colorbar()
    plt.savefig(f"div_v_star_{DIM}d_{N}.png")
    plt.close()
    return rhs

def vis_pressure(frame, masked=False):
    file_sol = os.path.join(example_folder, f"pressure_{frame}.bin")
    sol = load_vector(file_sol)
    if masked: sol = abs(sol) > 1e-16
    plt.imshow(sol.reshape((N,)*DIM).T, origin='lower')
    plt.colorbar()
    plt.savefig(f"pressure_{DIM}d_{N}.png")
    plt.close()
    return sol

if __name__ == '__main__':
    frame = 200
    vis_flags(frame)
    rhs = vis_div_v(frame, masked=True)
    sol = vis_pressure(frame, masked=False)