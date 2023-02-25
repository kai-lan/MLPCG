import numpy as np
import matplotlib.pyplot as plt

# dcdm_loss = "data_dcdm/output_2D_64/training_loss_empty.npy"
fluidnet_loss = "data_fluidnet/_output_2D_64/training_loss_smoke.npy"
# dcdm = np.load(dcdm_loss)
fluidnet = np.load(fluidnet_loss)
# plt.plot(dcdm, label='dcdm')
plt.plot(fluidnet, label='fluidnet')
plt.legend()
plt.xlabel("Training epoch")
plt.ylabel("Training loss")
plt.savefig("training_loss_smoke_with_bd.png", bbox_inches='tight')