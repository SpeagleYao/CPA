from img_aug import data_generator
from models import *
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# model = ResNet34().cuda()
# model.load_state_dict(torch.load('./pth/ResNet34.pth'))
model = VGG16().cuda()
model.load_state_dict(torch.load('./pth/VGG16.pth'))

with torch.no_grad():
    model.eval()
    criterion_mse = nn.MSELoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()
    # g_val = data_generator('./data/img_train_3d_224.npy', './data/tar_train_3d_224.npy',400 , train=False)
    g_val = data_generator('./data/img_test.npy', './data/tar_test.npy', 200, train=False)
    img, tar = g_val.gen()
    img = img.cuda()
    tar = tar.cuda()
    out = model(img)
    loss_mse = criterion_mse(out, tar)
    loss_l1 = criterion_l1(out, tar)
    print("Loss_mse:{0}".format(format(loss_mse, ".6f")))
    print("Loss_l1:{0}".format(format(loss_l1, ".6f")))

out = out.cpu().numpy()
tar = tar.cpu().numpy()

plt.scatter(out, tar, marker='.')
plt.plot([0, 0.9], [0, 0.9], 'r')
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.title("L1    Loss_l1:" + str(round(float(loss_l1.cpu()), 5)) + "    Loss_mse:" + str(round(float(loss_mse.cpu()), 5)))
# plt.title("Prediction V.S. True Label")
plt.savefig('./img_result/out_tar_wore.png')

# for i in range(10):
#     print('Val %2d: out=%.4f\ttar=%.4f' % (i, out[i], tar[i]))
