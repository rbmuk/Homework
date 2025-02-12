import numpy as np

n = 5;
M = np.random.randint(-10, 11, size=(n, n))
M = M + M.T
P = np.random.randint(-10, 11, size=(n, n))
Q = np.matmul(P, np.matmul(M, P.T))
print("Eigenvectors of M: ")
for v in np.linalg.eig(M).eigenvectors:
    print(v / np.linalg.norm(v))
for v in np.linalg.eig(Q):
    print(v / np.linalg.norm(v))
