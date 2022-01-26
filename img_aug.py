import cv2
import numpy as np
import Augmentor
import shutil
import random
import torch
from torchvision.transforms import transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class data_generator():
    def __init__(self, img_dir, tar_dir, batch_size = 128, train=False, seed=None):
        super(data_generator, self).__init__()

        data = np.expand_dims(np.load(img_dir), axis=1)
        label = np.expand_dims(np.load(tar_dir), axis=1)
        # print(data.shape, label.shape)

        self.p = Augmentor.DataPipeline(data, label.tolist())

        if train:
            pass
            self.p.rotate(probability=1.0, max_left_rotation=25, max_right_rotation=25)
            self.p.flip_left_right(probability=0.5)
            self.p.flip_top_bottom(probability=0.5)
            # self.p.skew(probability=0.25)
            # self.p.random_distortion(probability=0.25, grid_height=16, grid_width=16, magnitude=1)
            self.p.crop_random(probability=0.5, percentage_area=0.9)
            # self.p.resize(probability=1, width=512, height=512)
            # self.p.resize(probability=1, width=224, height=224)
        else:
            seed = 1105

        self.p.resize(probability=1, width=224, height=224)

        if seed:
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

        self.g = self.p.generator(batch_size)

    def gen(self):
        img_aug, y = next(self.g)
        img_aug = torch.as_tensor(np.array(img_aug)).float().detach().requires_grad_(True)/255
        y = torch.as_tensor(np.array(y)).float().detach().requires_grad_(True)
        # print(img_aug.shape, len(y))
        # img_tot = np.array(next(self.g))
        # img_aug = np.expand_dims(img_tot[:,0,:,:], axis=1)/255
        # gt_aug = np.expand_dims(img_tot[:,1,:,:], axis=1)/255
        # img_aug = torch.from_numpy(img_aug).float().detach().requires_grad_(True)
        # gt_aug = torch.from_numpy(gt_aug).float().detach().requires_grad_(True)

        return img_aug, y

if __name__=='__main__':
    img_dir = './data/img_train.npy'
    tar_dir = './data/tar_train.npy'
    g = data_generator(img_dir, tar_dir, batch_size=128, train=True)
    img, tar = g.gen()
    img = img.cuda()
    tar = tar.cuda()
    print(img.shape, tar.shape) # bs, 1, 512, 512
    print(img[0].max(), img[0].min(), img[0].mean(), img[0].std())
    # ind = 0
    # a = np.hstack((img.detach().numpy()[ind][0]*255, tar.detach().numpy()[ind][0]*255))
    # cv2.imwrite('testimage.png', a)
    # cv2.imshow('data', img.detach().numpy()[ind][0]*255)
    # cv2.waitKey(0)
