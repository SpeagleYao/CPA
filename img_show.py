import cv2
import numpy as np

src_img = np.load('./data/data_3df0.npy')
src_lab = np.load('./data/label_3df0.npy')
print(src_img.shape)

ind = 100
# cv2.imshow('Imgae '+str(ind)+' CPA: '+str(src_lab[ind]), src_img[ind])
# cv2.waitKey(0)
cv2.imwrite('./img/Imgae '+str(ind)+' CPA: '+str(src_lab[ind]) + '.jpg', src_img[ind])