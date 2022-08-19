from img_aug import data_generator
from models import *
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


path = './pth_BS16/VGG16_'
model_names = ["1.pth", "2.pth", "3.pth"]

mse_test = np.zeros(3)
l1_test = np.zeros(3)

ind = 0

for name in model_names:
    model = VGG16().cuda()
    model.load_state_dict(torch.load(path+name))

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

    mse_test[ind] = loss_mse
    l1_test[ind] = loss_l1
    ind += 1

print("MSE: %.6f ± %.6f" % (np.mean(mse_test), np.std(mse_test)))
print("L1: %.6f ± %.6f" % (np.mean(l1_test), np.std(l1_test)))

    

# out = out.cpu().numpy()
# tar = tar.cpu().numpy()

# for i in range(10):
#     print('Val %2d: out=%.4f\ttar=%.4f' % (i, out[i], tar[i]))






# from img_aug import data_generator
# from models import *
# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from matplotlib import pyplot as plt

# # model = ResNet34().cuda()
# # model.load_state_dict(torch.load('./pth/ResNet34.pth'))
# model = VGG16().cuda()
# model.load_state_dict(torch.load('./pth/VGG16_best.pth'))

# with torch.no_grad():
#     model.eval()
#     criterion_mse = nn.MSELoss().cuda()
#     criterion_l1 = nn.L1Loss().cuda()
#     # g_val = data_generator('./data/img_train_3d_224.npy', './data/tar_train_3d_224.npy',400 , train=False)
#     g_val = data_generator('./data/img_val_3d_224.npy', './data/tar_val_3d_224.npy', 200, train=False)
#     img, tar = g_val.gen()
#     img = img.cuda()
#     tar = tar.cuda()
#     out = model(img)
#     loss_mse = criterion_mse(out, tar)
#     loss_l1 = criterion_l1(out, tar)
#     print("Loss_mse:{0}".format(format(loss_mse, ".6f")))
#     print("Loss_l1:{0}".format(format(loss_l1, ".6f")))

# out = out.cpu().numpy()
# tar = tar.cpu().numpy()

# plt.scatter(out, tar, marker='.')
# plt.plot([0, 0.5], [0, 0.5], 'r')
# plt.xlabel("Prediction")
# plt.ylabel("True Label")
# plt.title("Loss_l1: " + str(loss_l1.cpu().numpy()))
# # plt.title("Prediction V.S. True Label")
# plt.savefig('./img_result/out_tar.png')

# # for i in range(10):
# #     print('Val %2d: out=%.4f\ttar=%.4f' % (i, out[i], tar[i]))
