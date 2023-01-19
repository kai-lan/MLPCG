import numpy as np
import matplotlib.pyplot as plt
import os

dim = 64
DIM = 2
path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(path, "..", "data_dcdm", f"output_{dim}_{DIM}D")

loss_train = np.load(data_path + "/training_loss_Wed-Jan-18-20:34:43-2023.npy")
loss_test = np.load(data_path + "/validation_loss_Wed-Jan-18-20:34:43-2023.npy")

fig, axes = plt.subplots(2, 1)
axes[0].plot(loss_train, label="train")
axes[1].plot(loss_test, label="validation")
plt.xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[1].set_ylabel("Loss")
axes[0].set_title("Training")
axes[1].set_title("Validation")
plt.savefig(data_path + "/loss.png", bbox_inches="tight")