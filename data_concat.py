import numpy as np

d3 = np.load("./data/data_3d3_224.npy")
d4 = np.load("./data/data_3d4_224.npy")
df0 = np.load("./data/data_3df0_224.npy")
df1 = np.load("./data/data_3df1_224.npy")

d = np.concatenate((d3, d4, df0, df1))
print(d.shape)
np.save('./data/data_3d_224.npy', d)

l3 = np.load("./data/label_3d3.npy")
l4 = np.load("./data/label_3d4.npy")
lf0 = np.load("./data/label_3df0.npy")
lf1 = np.load("./data/label_3df1.npy")

l = np.concatenate((l3, l4, lf0, lf1))
print(l.shape)
np.save('./data/label_3d.npy', l)