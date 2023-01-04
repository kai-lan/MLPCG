import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
#import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../dataset_mlpcg")
sys.path.insert(1, os.path.join(dir_path, '../lib'))
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

dim = 64
dim2 = dim**3
lr = 1.0e-4

# command variables
epoch_num = 50
b_size = 10 # batch size
epoch_sub_train = 10 # Number of epochs for each sub-training
size_sub_train = 1000 # Number of vectors for sub-training
# Use CUDA for training
cuda = torch.device("cuda")

name_sparse_matrix = os.path.join(data_path, "original_matA/A_origN64.bin")
A_sparse_scipy = hf.readA_sparse(dim, name_sparse_matrix,'f')

CG = cg.ConjugateGradientSparse(A_sparse_scipy)

coo = A_sparse_scipy.tocoo()

indices = np.mat([coo.row, coo.col])
# A_sparse = torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float32, device=cuda)
A_sparse = torch.sparse_csr_tensor(A_sparse_scipy.indptr, A_sparse_scipy.indices, A_sparse_scipy.data, A_sparse_scipy.shape, dtype=torch.float32, device=cuda)

class DCDM(nn.Module):
    def __init__(self):
        super(DCDM, self).__init__()
        self.cnn1 = nn.Conv3d(1, 16, 3, padding='same')
        self.cnn2 = nn.Conv3d(16, 16, 3, padding='same')
        self.cnn3 = nn.Conv3d(16, 16, 3, padding='same')
        self.cnn4 = nn.Conv3d(16, 16, 3, padding='same')
        self.cnn5 = nn.Conv3d(16, 16, 3, padding='same')

        self.downsample = nn.AvgPool3d(2)
        self.down1 = nn.Conv3d(16, 16, 3, padding='same')
        self.down2 = nn.Conv3d(16, 16, 3, padding='same')
        self.down3 = nn.Conv3d(16, 16, 3, padding='same')
        self.down4 = nn.Conv3d(16, 16, 3, padding='same')
        self.down5 = nn.Conv3d(16, 16, 3, padding='same')
        self.down6 = nn.Conv3d(16, 16, 3, padding='same')
        self.upsample = nn.Upsample(scale_factor=2)

        self.cnn6 = nn.Conv3d(16, 16, 3, padding='same')
        self.cnn7 = nn.Conv3d(16, 16, 3, padding='same')
        self.cnn8 = nn.Conv3d(16, 16, 3, padding='same')
        self.cnn9 = nn.Conv3d(16, 16, 3, padding='same')
        self.dense = nn.Linear(16, 1) # dim x dim x dim x nc -> dim x dim x dim x 1
    def forward(self, x): # shape: bs x nc x dim x dim x dim
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
        last_layer = self.dense(upa.permute(0, 2, 3, 4, 1)) # bs x nc x dim x dim x dim -> bs x dim x dim x dim x nc
        last_layer = last_layer.squeeze(-1)
        return last_layer

class CustomLossCNN1DFast(nn.Module):
    def __init__(self):
        super(CustomLossCNN1DFast, self).__init__()
    def forward(self, y_pred, y_true): # bs x dim x dim x dim
        ''' y_true: r, (bs, N)
        y_pred: A_hat^-1 r, (bs, N)
        '''
        y_pred = y_pred.flatten(1) # Keep bs
        y_true = y_true.flatten(1)
        YhatY = (y_true * y_pred).sum(dim=1) # Y^hat * Y, (bs,)
        YhatAt = (A_sparse @ y_pred.T).T # y_pred @ A_sparse.T not working, Y^hat A^T, (bs, N)
        YhatYhatAt = (y_pred * YhatAt).sum(dim=1) # Y^hat * (Yhat A^T), (bs,)
        return (y_true - torch.diag(YhatY/YhatYhatAt) @ YhatAt).square().sum(dim=1).mean()

class MyDataset(Dataset):
    def __init__(self, data_folder, dim=64, transform=None):
        self.data_folder = data_folder
        self.dim = dim
        self.transform = transform
    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.data_folder + f"/b_{index}.npy"))
        x = x.view(dim, dim, dim)
        if self.transform is not None: x = self.transform(x)
        return x
    def __len__(self):
        return len(os.listdir(self.data_folder))

model = DCDM()
model.to(cuda)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = CustomLossCNN1DFast()

# testing data rhs
# rand_vec_x = np.random.normal(0, 1, [dim2])
# b_rand = CG.multiply_A_sparse(rand_vec_x)

# test_folder = data_path +  "/test_matrices_and_vectors/N64/"
# b_smoke = hf.get_frame_from_source(10, test_folder + "smoke_passing_bunny")

training_loss = []
validation_loss = []
time_history = []
total_data_points = 20000
iters_sub_train = round(total_data_points/size_sub_train)
data_set = MyDataset(os.path.join(data_path, f"train_{dim}_3D"))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=size_sub_train, shuffle=True)

# Traing
for i in range(1, epoch_num):
    print(f"Training at {i} / {epoch_num}")
    t0 = time.time()
    tot_loss_train, tot_loss_val = 0, 0
    for ii, data in enumerate(data_loader, 1):
        data = data.to(cuda)
        print(f"Sub training at {ii} / {iters_sub_train} at training {i}")
        sub_train_size = round(0.9 * len(data))
        train_loader = torch.utils.data.DataLoader(data[:sub_train_size], batch_size=b_size)
        test_loader = torch.utils.data.DataLoader(data[sub_train_size:], batch_size=b_size)
        for j in range(epoch_sub_train): # One epoch at sub training
            epoch_loss = 0
            for jj, x in enumerate(train_loader, 1): # x: (bs, dim, dim, dim)
                y_pred = model(x.unsqueeze(1)) # input: (bs, 1, dim, dim, dim)
                loss = loss_fn(y_pred, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(j, epoch_loss)
            tot_loss_train += epoch_loss
        with torch.no_grad():
            loss = 0
            for jj, x in enumerate(test_loader, 1):
                y_pred = model(x.unsqueeze(1)) # input: (bs, 1, dim, dim, dim)
                loss += loss_fn(y_pred, x).item()
            tot_loss_val += loss
        print("Inner", tot_loss_train, tot_loss_val)
    tot_loss_train /= iters_sub_train
    tot_loss_val /= iters_sub_train
    training_loss.append(tot_loss_train)
    validation_loss.append(tot_loss_val)
    time_history.append(time.time())

outdir = os.path.join(data_path + f"output_{dim}_3D")
os.makedirs(outdir, exist_ok=True)
np.save(os.path.join(outdir, "training_loss.npy"), training_loss)
np.save(os.path.join(outdir, "validation_loss.npy"), validation_loss)
np.save(os.path.join(outdir, "time.npy"), time_history)
# Save model
torch.save(model.state_dict(), os.path.join(outdir, "model.pth"))
exit()
b_rhs = np.zeros((loading_number, dim2))
for i in range(1,epoch_num):
    print("Training at i = " + str(i))

    training_loss_inner = []
    validation_loss_inner = []
    t0=time.time()
    perm = np.random.permutation(total_data_points)
    for ii in range(for_loading_number):
        print("Sub_training at ",ii,"/",for_loading_number," at training ",i)


        # Loasing the data
        for j in range(loading_number):
            with open(foldername+str(perm[loading_number*ii+j])+'.npy', 'rb') as f:
                b_rhs[j] = np.load(f)

        sub_train_size = round(0.9*loading_number)
        sub_test_size = loading_number - sub_train_size
        iiln = ii*loading_number
        x_train = tf.convert_to_tensor(b_rhs[0:loading_number].reshape([loading_number,dim,dim,dim,1]),dtype=tf.float32)
        x_test = tf.convert_to_tensor(b_rhs[sub_train_size:loading_number].reshape([sub_test_size,dim,dim,dim,1]),dtype=tf.float32)

        hist = model.fit(x_train,x_train,
                        epochs=epoch_each_iter,
                        batch_size=b_size,
                        shuffle=True,
                        validation_data=(x_test,x_test))

        training_loss_inner = training_loss_inner + hist.history['loss']
        validation_loss_inner = validation_loss_inner + hist.history['val_loss']

    time_cg_ml = (time.time() - t0)
    print("Training loss at i = ",sum(training_loss_inner)/for_loading_number)
    print("Validation loss at i = ",sum(training_loss_inner)/for_loading_number)
    print("Time for epoch = ",i," is ", time_cg_ml)
    training_loss = training_loss + [sum(validation_loss_inner)/for_loading_number]
    validation_loss = validation_loss + [sum(validation_loss_inner)/for_loading_number]

    os.system("mkdir ./saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i))
    os.system("touch ./saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/model.json")
    model_json = model.to_json()
    model_name_json = project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/"
    with open(model_name_json+ "model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name_json + "model.h5")

    with open(training_loss_name, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name, 'wb') as f:
        np.save(f, np.array(validation_loss))
    print(training_loss)
    print(validation_loss)

    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
    max_it=30
    tol=1.0e-12

    print("Rotating Fluid Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_rotate, np.zeros(b_rotate.shape), model_predict, max_it,tol, True)
    print("Smoke Plume Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_smoke, np.zeros(b.shape), model_predict, max_it,tol, True)
    print("RandomRHSi Test")
    x_sol, res_arr_ml_generated_cg = CG.dcdm(b_rand, np.zeros(b_rand.shape), model_predict, max_it,tol, True)



