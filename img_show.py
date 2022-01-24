import cv2
import numpy as np

src_img = np.load('./data/data.npy')
src_lab = np.load('./data/label.npy')
print(src_img.shape)

ind = 0
cv2.imshow('Imgae '+str(ind)+' CPA: '+str(src_lab[ind]), src_img[ind])
cv2.waitKey(0)