import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, shape, image, suffix=''):
        self.data_folder = data_folder
        self.perm = permutation
        self.shape = shape
        self.iamge = image
        self.suffix = suffix
    def __getitem__(self, index):
        index = self.perm[index]
        x = torch.stack([
            torch.load(f"{self.data_folder}/b_{index}{self.suffix}.pt"),
            torch.load(f"{self.data_folder}/{self.iamge}{self.suffix}.pt")
        ]).reshape(self.shape)
        return x
    def __len__(self):
        return len(self.perm)
