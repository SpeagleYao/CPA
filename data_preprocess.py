import cv2
import os
import nrrd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageOps


file_num = '2_92'
filepath = '../Code/' + file_num + '/'
outpath = '../Seg_image/' + file_num + '/'
# 1_1  1_2  2_92  3_9
# 97   154  90    9

imgList = os.listdir(filepath)  # 读取工程文件夹下存放图片的文件夹的图片名
imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片名

list_len = int(len(imgList)/3)
img_ind = []

for i in range(0, list_len):
    x = re.findall('\d+', imgList[i*3])[0]
    if file_num == '2_92' and (x == '142' or x == '144'): continue
    img_ind.append(x)
# print(len(img_ind), img_ind)

srcimg = None
for i in tqdm(range(0, len(img_ind))): 
    seg_filename = filepath + img_ind[i] + '.seg.nrrd'
    ori_filename = filepath + img_ind[i] + '.nrrd'
    seg_data, _ = nrrd.read(seg_filename, index_order='C')
    for j in range(seg_data.shape[0]):
        if(seg_data[j].max()):
            ori_data, _ = nrrd.read(ori_filename, index_order='C')
            # ori_data = normalization(ori_data) * 255
            # print(ori_data.max(), ori_data.min())
            newimg = ori_data[j] * seg_data[j]
            # newimg = normalization(newimg) * 255
            # newimg = newimg * seg_data[j]
            # newimg = normalization(newimg) * 255
            # print(newimg.max(), newimg.min())
            newimg = np.asarray(Image.fromarray(newimg).convert('L'))
            newimg = newimg.reshape(1, 512, 512)
            if i==0:
                srcimg = newimg
            else:
                srcimg = np.append(srcimg, newimg, axis=0)
            break
    # if i == 5: break

print(srcimg.shape)
np.save("./data/data_" + file_num + ".npy", srcimg)

# outimg = Image.fromarray(srcimg).convert('RGB')
# outimg = Image.fromarray(srcimg).convert('L')
# outimg = ImageOps.autocontrast(outimg)
# outimg = ImageOps.equalize(outimg)
# outimg.show()
# a = np.asarray(outimg)
# print(a.max(), a.min())
# plt.hist(a.ravel(), 256, [1, 256], color='r')
# plt.show()
# outimg.save(outpath + '1.jpg')

# plt.hist(srcimg[0].ravel(), 256, [0, 256], color='r')
# plt.hist(cv2.equalizeHist(srcimg[0]).ravel(), 256, [0, 256], color='r')
# plt.show()
# cv2.equalizeHist

# hist = cv2.calcHist([srcimg[0]],[0],None,[256],[0,255])
# plt.plot(hist,'r')
# plt.show()

# print(srcimg.shape)
# scio.savemat('Ori_img.mat', {'img':srcimg})
# np.save("Ori_data.npy", srcimg)
#***************************************************************