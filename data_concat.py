import numpy as np

d1_1 = np.load("./data/image/data_1_1.npy")
d1_2 = np.load("./data/image/data_1_2.npy")
d2_92 = np.load("./data/image/data_2_92.npy")
d3_9 = np.load("./data/image/data_3_9.npy")

d = np.concatenate((d1_1, d1_2, d2_92, d3_9))
print(d.shape)
np.save('./data/data.npy', d)

l1_1 = np.load("./data/CPA/label_1_1.npy")
l1_2 = np.load("./data/CPA/label_1_2.npy")
l2_92 = np.load("./data/CPA/label_2_92.npy")
l3_9 = np.load("./data/CPA/label_3_9.npy")

l = np.concatenate((l1_1, l1_2, l2_92, l3_9))
print(l.shape)
np.save('./data/label.npy', l)