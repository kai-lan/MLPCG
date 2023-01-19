'''
File: dataset_load.py
File Created: Tuesday, 10th January 2023 8:21:06 pm

Author: Kai Lan (kai.weixian.lan@gmail.com)
Last Modified: Tuesday, 10th January 2023 9:27:00 pm
--------------
'''
import torch
from torch.utils.data import Dataset, DataLoader
from GLOBAL_VARS import *

class FluidNetDataset(Dataset):
    """Fluid Net dataset."""

    def __init__(self, num_images, num_frames, training=True):
        self.num_images = num_images
        self.num_frames = num_frames
        if training: self.data_path = DATA_TR_2D_PATH__
        else:        self.data_path = DATA_TE_2D_PATH__
    def __len__(self):
        return self.num_frames * self.num_images

    # Only load rhs, no gt(pressure) loaded
    def __getitem__(self, idx):
        image = idx // self.num_frames
        frame = idx % self.num_frames
        assert image < self.num_images, f"Image number {image} or frame number {frame} exceeds the capacity {self.num_images} and {self.num_frames}."
        file_div = os.path.join(self.data_path, f"{image:06}", f"{frame*FRAME_INCREMENT__:06}_div.pt")
        file_flags = os.path.join(self.data_path, f"{image:06}", f"{frame*FRAME_INCREMENT__:06}_flags.pt")
        assert os.path.exists(file_div), file_div
        assert os.path.exists(file_flags), file_flags
        div_U = torch.load(file_div).squeeze(0)
        flags = torch.load(file_flags).squeeze(0)
        return torch.cat([div_U, flags], dim=0)

if __name__ == '__main__':
    dataset = FluidNetDataset(20, 40)
    loader = DataLoader(dataset, batch_size=5)
    for x in loader:
        print(x.shape)
        break