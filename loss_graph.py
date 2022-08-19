import numpy as np
from matplotlib import pyplot as plt
import matplotlib

loss = np.load('./loss/loss.npy')
loss_val = np.load('./loss/loss_val.npy')

epoch = np.arange(0, loss.shape[0])
# print(epoch)
# print(loss.shape, epoch.shape, loss_val.shape)
plt.plot(epoch[::25], loss[::25], label="Train Loss", linewidth = 1)
plt.plot(epoch[::25], loss_val[::25], label="Val Loss", linewidth = 1)
plt.legend(loc='upper right')
plt.title('Loss Graph')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./loss_graph/loss.png')