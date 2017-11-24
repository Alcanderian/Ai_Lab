import numpy as np

Xy = np.loadtxt('../../Data/littletrain.csv', delimiter=',')
X, y = Xy[:, :-1], Xy[:, -1:]
N, M = X.shape
X, w = np.column_stack((np.ones((N, 1)), X)), np.ones((M + 1, 1))

for i in range(1):
    w -= 1.0 * (np.dot(X.T, 1.0 / (1.0 + np.exp(-np.dot(X, w))) - y) + 0.0 * np.append([[0.0]], w[:, 1:])) / N

p = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
p[p > 0.5], p[p <= 0.5] = 1, 0
print(w)
print('Accu: ' + str(float(sum(p == y)) / N))
