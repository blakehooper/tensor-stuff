import numpy as np

wines = np.genfromtxt("winequality-red.csv", delimiter=";", skip_header=1)

wines = np.array(wines[1:], dtype=np.float)
print(wines[:3])
print(wines.shape)

random = np.random.rand(3, 4)
print(random)
print(random[:3,3])