from PIL.Image import new
import nrrd
import numpy as np
import cv2

srcimg = None
for i in range(1, 101):
    if i==79 or i==84 or i==91: continue
    nrrd_filename = '1_1/'+str(i)+'.seg.nrrd'
    nrrd_data, nrrd_options = nrrd.read(nrrd_filename, index_order='C')
    for j in range(nrrd_data.shape[0]):
        if(nrrd_data[j].max()):
            newimg = nrrd_data[j].reshape(1, 512, 512)
            if i==1:
                srcimg = newimg
            else:
                srcimg = np.append(srcimg, newimg, axis=0)
print(srcimg.shape)
np.save("Seg_data.npy", srcimg)
# nrrd_filename = '1_1/8.seg.nrrd'
# nrrd_data, nrrd_options = nrrd.read(nrrd_filename, index_order='C')

# seg_index = []

# print(nrrd_data.shape)

# for i in range(nrrd_data.shape[0]):
#     if(nrrd_data[i].max()): seg_index.append(i)

# print(seg_index)
# cv2.imshow('seg_image', nrrd_data[seg_index[0]]*255)
# cv2.waitKey(0)
