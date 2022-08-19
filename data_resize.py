import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

data = np.load('./data/data_3df0.npy')
print(data.shape)

new_size = (224, 224)

img = data[0]
cub_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

for i in tqdm(range(data.shape[0])):

    img = data[i]
    newimg = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    newimg = np.asarray(Image.fromarray(newimg).convert('L')).reshape(1, 224, 224)
    if not 'srcimg' in dir():
        srcimg = newimg
    else:
        srcimg = np.append(srcimg, newimg, axis=0)    

print(srcimg.shape)
data = np.save('./data/data_3df0_224.npy', srcimg)