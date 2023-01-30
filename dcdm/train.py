import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../data_dcdm")
sys.path.insert(1, os.path.join(dir_path, '../lib'))
import conjugate_gradient as cg
import read_data as hf

class DCDM(nn.Module):
    def __init__(self, DIM):
        super(DCDM, self).__init__()
        self.DIM = DIM
        Conv = eval(f"nn.Conv{DIM}d")
        AvgPool = eval(f"nn.AvgPool{DIM}d")
        self.cnn1 = Conv(1, 16, kernel_size=3, padding='same')
        self.cnn2 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn3 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn4 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn5 = Conv(16, 16, kernel_size=3, padding='same')

        self.downsample = AvgPool(2)
        self.down1 = Conv(16, 16, kernel_size=3, padding='same')
        self.down2 = Conv(16, 16, kernel_size=3, padding='same')
        self.down3 = Conv(16, 16, kernel_size=3, padding='same')
        self.down4 = Conv(16, 16, kernel_size=3, padding='same')
        self.down5 = Conv(16, 16, kernel_size=3, padding='same')
        self.down6 = Conv(16, 16, kernel_size=3, padding='same')
        self.upsample = nn.Upsample(scale_factor=2)

        self.cnn6 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn7 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn8 = Conv(16, 16, kernel_size=3, padding='same')
        self.cnn9 = Conv(16, 16, kernel_size=3, padding='same')
        self.dense = nn.Linear(16, 1) # dim x dim (x dim) x nc -> dim x dim (x dim) x 1
    def forward(self, x): # shape: bs x nc x dim x dim (x dim)
        first_layer = self.cnn1(x)
        la = F.relu(self.cnn2(first_layer))
        lb = F.relu(self.cnn3(la))
        la = F.relu(self.cnn4(lb) + la)
        lb = F.relu(self.cnn5(la))

        apa = self.downsample(lb)
        apb = F.relu(self.down1(apa))
        apa = F.relu(self.down2(apb) + apa)
        apb = F.relu(self.down3(apa))
        apa = F.relu(self.down4(apb) + apa)
        apb = F.relu(self.down5(apa))
        apa = F.relu(self.down6(apb) + apa)

        upa = self.upsample(apa) + lb
        upb = F.relu(self.cnn6(upa))
        upa = F.relu(self.cnn7(upb) + upa)
        upb = F.relu(self.cnn8(upa))
        upa = F.relu(self.cnn9(upb) + upa)
        if self.DIM == 3:
            last_layer = self.dense(upa.permute(0, 2, 3, 4, 1)) # bs x nc x dim x dim x dim -> bs x dim x dim x dim x nc
        else:
            last_layer = self.dense(upa.permute(0, 2, 3, 1)) # bs x nc x dim x dim -> bs x dim x dim x nc
        last_layer = last_layer.squeeze(-1)
        return last_layer

class CustomLossCNN1DFast(nn.Module):
    def __init__(self):
        super(CustomLossCNN1DFast, self).__init__()
    def forward(self, y_pred, y_true): # bs x dim x dim (x dim)
        ''' y_true: r, (bs, N)
        y_pred: A_hat^-1 r, (bs, N)
        '''
        y_pred = y_pred.flatten(1) # Keep bs
        y_true = y_true.flatten(1)
        YhatY = (y_true * y_pred).sum(dim=1) # Y^hat * Y, (bs,)
        YhatAt = (A_sparse @ y_pred.T).T # y_pred @ A_sparse.T not working, Y^hat A^T, (bs, N)
        YhatYhatAt = (y_pred * YhatAt).sum(dim=1) # Y^hat * (Yhat A^T), (bs,)
        return (y_true - torch.diag(YhatY/YhatYhatAt) @ YhatAt).square().sum(dim=1).mean() # /bs / N

class MyDataset(Dataset):
    def __init__(self, data_folder, permutation, bc, dim=64, DIM=2, transform=None):
        self.data_folder = data_folder
        self.dim = dim
        self.DIM = DIM
        self.bc = bc
        self.perm = permutation
        self.transform = transform
    def __getitem__(self, index):
        index = self.perm[index]
        x = torch.from_numpy(np.load(self.data_folder + f"/b_{self.bc}_{index}.npy"))
        x = x.view((dim,)*DIM)
        if self.transform is not None: x = self.transform(x)
        return x
    def __len__(self):
        return len(self.perm)
        # return len([file for file in os.listdir(self.data_folder) if file.startswith(f'b_{self.bc}')])

if __name__ == '__main__':
    import logging
    # command variables
    dim = 64
    DIM = 2
    bc = 'solid'
    dim2 = dim**DIM
    lr = 1.0e-4
    epoch_num = 10
    b_size = 25 # batch size, 3D data with big batch size (>50) cannot fit in GPU >-<
    total_data_points = 2000
    cuda = torch.device("cuda") # Use CUDA for training

    def log(name, value): return f"{name:<30}{value:<20}\n"
    info = "\n" + "Basic variables\n" + '-'*50 + '\n'
    info += log("N", dim)
    info += log("DIM", DIM)
    info += log("bc", bc)
    info += log("lr", lr)
    info += log("Epoches", epoch_num)
    info += log("batch size", b_size)
    info += log("Tot data points", total_data_points)

    name_sparse_matrix = os.path.join(data_path, f"train_{DIM}D_{dim}/A_{bc}.bin")
    A_sparse_scipy = hf.readA_sparse(dim, name_sparse_matrix, DIM, 'f')

    CG = cg.ConjugateGradientSparse(A_sparse_scipy)

    coo = A_sparse_scipy.tocoo()

    indices = np.mat([coo.row, coo.col])
    A_sparse = torch.sparse_csr_tensor(A_sparse_scipy.indptr, A_sparse_scipy.indices, A_sparse_scipy.data, A_sparse_scipy.shape, dtype=torch.float32, device=cuda)

    model = DCDM(DIM)
    model.to(cuda)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = CustomLossCNN1DFast()

    training_loss = []
    validation_loss = []
    time_history = []
    perm = np.random.permutation(total_data_points)
    train_size = round(0.8 * total_data_points)
    train_set = MyDataset(os.path.join(data_path, f"train_{DIM}D_{dim}"), perm[:train_size], bc, dim=dim, DIM=DIM)
    validation_set = MyDataset(os.path.join(data_path, f"train_{DIM}D_{dim}"), perm[train_size:], bc, dim=dim, DIM=DIM)
    train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=b_size, shuffle=False)

    # Traing
    for i in range(1, epoch_num+1):
        print(f"Training at {i} / {epoch_num}")
        t0 = time.time()
        tot_loss_train, tot_loss_val = 0, 0
        # Training
        for ii, x in enumerate(tqdm(train_loader), 1):# x: (bs, dim, dim, dim)
            x = x.to(cuda)
            y_pred = model(x.unsqueeze(1)) # input: (bs, 1, dim, dim, dim)
            loss = loss_fn(y_pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        with torch.no_grad():
            for x in train_loader:
                x = x.to(cuda)
                y_pred = model(x.unsqueeze(1)) # input: (bs, 1, dim, dim, dim)
                tot_loss_train += loss_fn(y_pred, x).item()
            for x in validation_loader:
                x = x.to(cuda)
                y_pred = model(x.unsqueeze(1)) # input: (bs, 1, dim, dim, dim)
                tot_loss_val += loss_fn(y_pred, x).item()
        training_loss.append(tot_loss_train)
        validation_loss.append(tot_loss_val)
        time_history.append(time.time())
        print(training_loss[-1], validation_loss[-1])

    outdir = os.path.join(data_path, f"output_{DIM}D_{dim}")
    suffix = time.ctime().replace(' ', '-')
    logging.basicConfig(filename=os.path.join(outdir, f"settings_{suffix}.log"), filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(info)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"training_loss_{suffix}.npy"), training_loss)
    np.save(os.path.join(outdir, f"validation_loss_{suffix}.npy"), validation_loss)
    np.save(os.path.join(outdir, f"time_{suffix}.npy"), time_history)
    # Save model
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{suffix}.pth"))
