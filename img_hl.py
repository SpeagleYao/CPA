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
# model = VGG16().cuda()
# model.load_state_dict(torch.load('./pth/VGG16_best_L1.pth'))

with torch.no_grad():
    g_val = data_generator('./data/img_val_3d_224.npy', './data/tar_val_3d_224.npy', 200, train=False)
    img, tar = g_val.gen()
    img = img.detach().numpy()
    tar = tar.detach().numpy()

    print(img.shape, tar.shape)

    for i in range(tar.shape[0]):
        if tar[i] > 0.4:
            cv2.imwrite('./img/img_high/CPA: ' + str(tar[i]) + '.jpg', img[i][0]*255)
        elif tar[i] < 0.005:
            cv2.imwrite('./img/img_low/CPA: ' + str(tar[i]) + '.jpg', img[i][0]*255)

# for i in range(10):
#     print('Val %2d: out=%.4f\ttar=%.4f' % (i, out[i], tar[i]))
