import numpy as np

f2 = open("./CPA_1_1.txt","r")
lines = f2.readlines()
a = []
for line in lines:
    a.append(float(line))
a = np.array(a)
# print(a.shape)
np.save('Seg_label.npy', a)
# print(a[97])
# 两个89