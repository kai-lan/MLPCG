import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data_folder, file_format, permutation, shape):
        self.data_folder = data_folder
        self.file_format = file_format
        self.perm = permutation
        self.shape = shape
    def __getitem__(self, index):
        index = self.perm[index]
        item = []
        for format in self.file_format:
            file = os.path.join(self.data_folder, format.replace('*', str(index)))
            if file.endswith('npy'):
                x = torch.from_numpy(np.load(file))
                x = x.view(self.shape)
                item.append(x)
            elif file.endswith('pt'):
                x = torch.load(file)
                if (np.prod(x.shape) == np.prod(self.shape)):
                    x = x.reshape(self.shape)
                item.append(x)
            else: raise Exception("File format can only be .npy or .pt")
        return item
    def __len__(self):
        return len(self.perm)
