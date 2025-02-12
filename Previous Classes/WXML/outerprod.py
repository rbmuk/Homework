import numpy as np
def Q(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1, -1], [-1, 1]])
    return np.block([[Q(n-1) + np.outer(np.ones(2**(n-1)), np.ones(2**(n-1))),Q(n-1) - np.outer(np.ones(2**(n-1)), np.ones(2**(n-1)))],
                     [Q(n-1) - np.outer(np.ones(2**(n-1)), np.ones(2**(n-1))), Q(n-1) + np.outer(np.ones(2**(n-1)), np.ones(2**(n-1)))]])
n = 4
Q = Q(n)
eigenvalues, eigenvectors = np.linalg.eigh(Q)
print(np.around(eigenvalues, 1).astype(int))
print(eigenvectors.T)
print(f'Q shape: {Q.shape}, rank: {np.linalg.matrix_rank(Q)}')