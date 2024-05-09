import numpy as np
from sklearn.model_selection import train_test_split

data1 = np.random.normal(size=(200, 3))
data2 = np.random.uniform(low=-5, high=5, size=(200, 3))
data3 = np.random.poisson(size=(200, 2))
data4 = np.random.randint(low=0, high=1, size=(200, 1))
y = np.random.randint(low=-30, high=30, size=(200, 1))
data_new = np.concatenate([data1, data2, data3, data4, y], axis=1)

noise = np.random.normal(size=(200, 9))
data_new[:, :9] = data_new[:, :9] + noise
# print(data_new[:2, :])
train, test = train_test_split(data_new, test_size=0.1)

np.save('train.npy', train)
np.save('test.npy', test)
print('Data generate')
