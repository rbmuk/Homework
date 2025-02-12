import numpy as np

n,d,k = 500, 1000, 100

X = np.random.uniform(-10, 10, (n, d))
w = [j/k if j <= k else 0 for j in range(1, d+1)]
y = X @ w + np.random.normal(0, 1, (n,))

np.savetxt('data/uniform/X.csv', X, delimiter=',')
np.savetxt('data/uniform/y.csv', y, delimiter=',')