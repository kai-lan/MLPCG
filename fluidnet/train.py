import os, sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from model import FluidNet
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../data_fluidnet")
sys.path.insert(1, os.path.join(dir_path, '../lib'))

# import conjugate_gradient as cg
# import read_data as hf
class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, N=64, DIM=2, transform=None):
        self.data_folder = data_folder
        self.N = N
        self.DIM = DIM
        self.perm = permutation
        self.transform = transform
    def __getitem__(self, index):
        index = self.perm[index]
        file_x = os.path.join(self.data_folder, "preprocessed", f"x_{index}.pt")
        file_A = os.path.join(self.data_folder, "preprocessed", f"A_{index}.pt")
        x = torch.load(file_x).reshape((2,) + (self.N,)*self.DIM)
        A = torch.load(file_A)
        if self.transform is not None: x = self.transform(x)
        return x, A
    def __len__(self):
        return len(self.perm)

class CustomLossCNN1DFast(nn.Module):
    def __init__(self):
        super(CustomLossCNN1DFast, self).__init__()
    def forward(self, x, y, A):
        bs = x.shape[0]
        r = torch.zeros(1)
        for i in range(bs):
            r += (y[i] - A[i] @ x[i]).norm() / y[i].norm()
        return r / bs

if __name__ == '__main__':
    import logging
    dim = 64
    DIM = 2
    dim2 = dim**DIM
    lr = 0.0001
    epoch_num = 100
    b_size = 10 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_iamges = 1000
    cuda = torch.device("cuda") # Use CUDA for training

    def log(name, value): return f"{name:<30}{value:<20}\n"
    info = "\n" + "Basic variables\n" + '-'*50 + '\n'
    info += log("N", dim)
    info += log("DIM", DIM)
    info += log("lr", lr)
    info += log("Epoches", epoch_num)
    info += log("batch size", b_size)
    info += log("Tot data points", total_iamges)

    # torch.manual_seed(2)
    model = FluidNet()
    model.to(cuda)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = CustomLossCNN1DFast()

    training_loss = []
    validation_loss = []
    time_history = []
    # perm = np.random.permutation(total_iamges) + 1
    perm = np.arange(total_iamges) + 1 # Debug sparse matrices

    train_size = round(0.8 * total_iamges)
    train_set = MyDataset(os.path.join(data_path, f"dambreak_{DIM}D_{dim}"), perm[:train_size], N=dim, DIM=DIM)
    validation_set = MyDataset(os.path.join(data_path, f"dambreak_{DIM}D_{dim}"), perm[train_size:], N=dim, DIM=DIM)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=b_size, shuffle=False)

    # Traing
    for i in range(1, epoch_num+1):
        print(f"Training at {i} / {epoch_num}")
        t0 = time.time()
        # Training
        los = 0.0
        for ii, (data, A) in enumerate(tqdm(train_loader), 1):# x: (bs, dim, dim, dim)
            data = data.to(cuda)
            A = A.to(cuda)
            x_pred = model(data) # (bs, 2, N, N)
            loss = loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            los += loss.item()

        # Validation
        tot_loss_train, tot_loss_val = 0, 0
        with torch.no_grad():
            for data, A in train_loader:
                data = data.to(cuda)
                A = A.to(cuda)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_train += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item()
            for data, A in validation_loader:
                data = data.to(cuda)
                A = A.to(cuda)
                x_pred = model(data) # input: (bs, 1, dim, dim)
                tot_loss_val += loss_fn(x_pred.squeeze(dim=1).flatten(1), data[:, 0].flatten(1), A).item() * 4
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time())
        print(training_loss[-1], validation_loss[-1])

    outdir = os.path.join(data_path, f"output_{DIM}D_{dim}")
    os.makedirs(outdir, exist_ok=True)
    suffix = time.ctime().replace(' ', '-')
    logging.basicConfig(filename=os.path.join(outdir, f"settings_{suffix}.log"), filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(info)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    # Save model
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))
