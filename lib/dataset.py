import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, shape):
        self.data_folder = data_folder
        self.perm = permutation
        self.shape = shape
    def __getitem__(self, index):
        index = self.perm[index]
        x = torch.stack([
            torch.load(f"{self.data_folder}/b_{index}.pt"),
            torch.load(f"{self.data_folder}/flags.pt")
        ]).reshape(self.shape)
        return x
    def __len__(self):
        return len(self.perm)
