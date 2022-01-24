import numpy as np

file_num = '3_9'

f2 = open("./data/CPA/CPA_" + file_num + ".txt","r")
lines = f2.readlines()
a = []
for line in lines:
    a.append(float(line))
a = np.array(a)
print(a.shape)
np.save("./data/CPA/label_" + file_num + ".npy", a)