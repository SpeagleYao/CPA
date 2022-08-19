import cv2
import numpy as np
import Augmentor
import shutil
import random
import torch
from torchvision.transforms import transforms
import os
from matplotlib import pyplot as plt

tar = np.load('./data/label_3d.npy')

plt.hist(tar, bins=20, facecolor="blue", edgecolor="black")
plt.savefig('tar_hist.jpg')