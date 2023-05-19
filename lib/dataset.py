import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, transform, suffix=''):
        self.data_folder = data_folder
        self.perm = permutation
        self.transform = transform
        self.suffix = suffix
    def __getitem__(self, index):
        index = self.perm[index]
        if self.suffix == 'mixed':
            suffix = np.random.choice(['', '_rand'])
            x = torch.load(f"{self.data_folder}/b_{index}{suffix}.pt")
        elif self.suffix == 'rhs':
            x = torch.load(f"{self.data_folder}/rhs.pt")
        else:
            x = torch.load(f"{self.data_folder}/b_{index}{self.suffix}.pt")
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
