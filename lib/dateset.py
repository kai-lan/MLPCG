import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from read_data import read_flags, load_vector

dir_path = os.path.dirname(os.path.realpath(__file__))

class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, N=64, DIM=2, transform=None):
        self.data_folder = data_folder
        self.N = N
        self.DIM = DIM
        self.perm = permutation
        self.transform = transform
    def __getitem__(self, index):
        index = self.perm[index]
        file_rhs = os.path.join(data_folder, f"div_v_star_{index}.bin")
        file_flags = os.path.join(data_folder, f"flags_{index}.bin")
        rhs = torch.tensor(load_vector(file_rhs), dtype=torch.float32).reshape((1,) + (self.N,)*self.DIM)
        flags = torch.tensor(read_flags(file_flags), dtype=torch.float32).reshape((1,) + (self.N,)*self.DIM)
        x = torch.cat([rhs, flags])
        if self.transform is not None: x = self.transform(x)
        return x
    def __len__(self):
        return len(self.perm)

if __name__ == '__main__':
    import numpy as np
    num_data = 300
    DIM = 2
    N = 64
    data_folder = os.path.join(dir_path,  "..", "data_fluidnet", f"dambreak_{DIM}D_{N}")
    perm = np.random.permutation(num_data)
    dataset = MyDataset(data_folder, perm, 64, 2)
    print(dataset[4].shape)