import numpy as np
import random

data = np.load('./data/data_3d_224.npy')
label = np.load('./data/label_3d.npy')

l_001 = label > 0.01
l_002 = label > 0.02
l_005 = label > 0.05

data001 = data[l_001]
label001 = label[l_001]
data002 = data[l_002]
label002 = label[l_002]
data005 = data[l_005]
label005 = label[l_005]

print(data001.shape, label001.shape)
print(data002.shape, label002.shape)
print(data005.shape, label005.shape)

np.save('./data/data001_3d_224.npy', data001)
np.save('./data/label001_3d_224.npy', label001)
np.save('./data/data002_3d_224.npy', data002)
np.save('./data/label002_3d_224.npy', label002)
np.save('./data/data005_3d_224.npy', data005)
np.save('./data/label005_3d_224.npy', label005)