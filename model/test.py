import numpy as np
y = np.loadtxt("data.txt", delimiter=',')[:, -1]
print("Labels found in data:", np.unique(y))
