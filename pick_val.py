import numpy as np
import random

data = np.load('./data/data.npy')
label = np.load('./data/label.npy')

data_len = data.shape[0]

x = random.sample(range(0, data_len), int(data_len/10))
y = np.ones(data_len)

for i in range(len(x)):
    y[x[i]] = 0
y = np.nonzero(y)

img_train = data[y]
tar_train = label[y]
img_val = data[x]
tar_val = label[x]

print(img_train.shape, img_val.shape)
print(tar_train.shape, tar_val.shape)

np.save('./data/img_train.npy', img_train)
np.save('./data/img_val.npy', img_val)
np.save('./data/tar_train.npy', tar_train)
np.save('./data/tar_val.npy', tar_val)