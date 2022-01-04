import cv2
import numpy as np

src_img = np.load('Seg_data.npy')
src_lab = np.load('Seg_label.npy')
print(src_img.shape)

ind = 7
cv2.imshow('Imgae '+str(ind)+' CPA: '+str(src_lab[ind]), src_img[ind]*255)
cv2.waitKey(0)