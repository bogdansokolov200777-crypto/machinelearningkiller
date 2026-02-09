import numpy as np
from sklearn.model_selection import train_test_split

numbers = np.random.randint(10, 101, (40,2))
binary = np.random.randint(0, 2, (40,1))
complexy = np.concatenate((numbers, binary), axis=1)

X = complexy[:, :2]
y = complexy[:, 2]

numbers_train, numbers_test, binary_train, binary_test = train_test_split(X, y, test_size=0.3, random_state=10)

print(f"Размер numbers_train: {numbers_train.shape}")
print(f"Размер numbers_test: {numbers_test.shape}")
print('123')