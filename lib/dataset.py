import torch
from torch.utils.data import Dataset
import numpy as np
from GLOBAL_VARS import *

class DDPDataset(Dataset):
    def __init__(self, permutation, DIM=3):
        self.perm = permutation
        self.suffix = f"_200_{DIM}D" if DIM == 3 else f"_200"

    def set_and_return_image(self, scene, frame, shape): # This will update for each frame
        self.data_folder = f"{DATA_PATH}/{scene}{self.suffix}/preprocessed/{frame}"
        self.shape = shape
        self.size = np.prod(shape)
        A = torch.load(f"{self.data_folder}/A.pt", map_location='cpu')
        image = torch.load(f"{self.data_folder}/flags_binary_3.pt", map_location='cpu')
        fluid_cells = torch.load(f"{self.data_folder}/fluid_cells.pt", map_location='cpu')
        self.fluid_cells = fluid_cells
        return A, image, fluid_cells

    def set_image(self, scene, frame, shape): # This will update for each frame
        self.data_folder = f"{DATA_PATH}/{scene}{self.suffix}/preprocessed/{frame}"
        self.shape = shape
        self.size = np.prod(shape)
        fluid_cells = torch.load(f"{self.data_folder}/fluid_cells.pt", map_location='cpu')
        self.fluid_cells = fluid_cells

    def __getitem__(self, index):
        index = self.perm[index]
        x = torch.load(f"{self.data_folder}/b_{index}.pt", map_location='cpu') # map to CPU first, then transfer to CUDA for the entrie batch: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        b = torch.zeros(self.size, dtype=torch.float32, device='cpu')
        b[self.fluid_cells] = x
        b = b.reshape(self.shape)
        return b

    def __len__(self):
        return len(self.perm)


class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, transform, suffix=''):
        self.data_folder = data_folder
        self.perm = permutation
        self.transform = transform
        self.suffix = suffix
    def __getitem__(self, index):
        index = self.perm[index]
        x = torch.load(f"{self.data_folder}/b_{index}{self.suffix}.pt", map_location='cuda') # map to CPU first, then transfer to CUDA for the entrie batch: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        x = self.transform(x)
        return x
    def __len__(self):
        return len(self.perm)

class RitzDataset(Dataset):
    def __init__(self, data, flags, perm, shape):
        self.data = data
        self.flags = flags
        self.perm = perm
        self.shape = shape
        self.fluidcells = torch.where(flags == 2)[0]
    def __getitem__(self, index):
        index = self.perm[index]
        b = torch.zeros(np.prod(self.shape)//2, dtype=torch.float32)
        b[self.fluidcells] = torch.from_numpy(self.data[index])
        x = torch.stack([
            b,
            self.flags
        ]).reshape(self.shape)
        return x
    def __len__(self):
        return len(self.perm)
