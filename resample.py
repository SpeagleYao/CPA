import numpy as np
import random

steps = 0.1
interval = 1/steps

img = np.load('./data/img_train_3d_224.npy')
tar = np.load('./data/tar_train_3d_224.npy')
x = tar

# x = np.random.rand(1000)
bins = np.arange(0, 0 + steps * interval, steps)
inds = np.digitize(x, bins)

# print(inds)
bin_len= bins.shape[0]
# print(bin_len)
num = np.zeros(bin_len)
for i in range(1, bin_len):
    num[i] += np.sum(inds == i)
    # print('%2d : %3d' % (i, num[i]))
    # print(bins[i-1], "<=", int(num[i]), "<", bins[i])
    print('%.1f <= X <= %.1f : %2d' % (bins[i-1], bins[i], num[i]))
print(img.shape, tar.shape)

for i in range(1, bin_len):
    a = np.argwhere(inds == i).squeeze(1)
    b = np.random.choice(a, size=500, replace=True)
    if not 'index' in dir():
        index = b
    else:
        index = np.append(index, b, axis=0)
print()
x = x[index]
inds = np.digitize(x, bins)
num = np.zeros(bin_len)
for i in range(1, bin_len):
    num[i] += np.sum(inds == i)
    # print('%2d : %3d' % (i, num[i]))
    print('%.1f <= X <= %.1f : %2d' % (bins[i-1], bins[i], num[i]))
tar = x
img = img[index]
print(img.shape, tar.shape)
np.save("./data/img_resample_224.npy", img)
np.save("./data/tar_resample_224.npy", tar)
