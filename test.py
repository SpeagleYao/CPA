from img_aug import data_generator
from models import *
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

model = VGG16()
model.load_state_dict(torch.load('./pth/VGG16.pth'))

model.eval()
criterion = nn.MSELoss()
g_val = data_generator('./data/img_train.npy', './data/tar_train.npy', 10, train=False)
# g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 35, train=False)
img, tar = g_val.gen()
# img = img.cuda()
# tar = tar.cuda()
out = model(img)
loss_val = criterion(out, tar)
print("Loss_val:{0}".format(format(loss_val, ".4f")))

for i in range(10):
    print('Val %2d: out=%.2f\ttar=%.2f' % (i, out[i], tar[i]))


# out = torch.where(out>=0.5, 1, 0)
# out = out.numpy().reshape(10, 224, 224)*255
# tar = tar.detach().numpy().reshape(10, 224, 224)*255
# for i in range(out.shape[0]):
#     a = np.hstack((tar[i], out[i]))
#     cv2.imwrite('./prdimg/prdimg'+str(i)+'.png', a)
