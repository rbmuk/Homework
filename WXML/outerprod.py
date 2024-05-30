import numpy as np
def Q(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1, -1], [-1, 1]])
    return np.block([[Q(n-1) + np.outer(np.ones(2**(n-1)), np.ones(2**(n-1))),Q(n-1) - np.outer(np.ones(2**(n-1)), np.ones(2**(n-1)))],
                     [Q(n-1) - np.outer(np.ones(2**(n-1)), np.ones(2**(n-1))), Q(n-1) + np.outer(np.ones(2**(n-1)), np.ones(2**(n-1)))]])
n = 4
eigenvalues, eigenvectors = np.linalg.eig(Q(n))
largest_eigenvalue = np.max(eigenvalues)
print("Largest eigenvalue:", largest_eigenvalue)
c = np.concatenate((np.ones(2**(n-2)), np.zeros(2**n - 2**(n-2)))).reshape(-1, 1)
c /= np.linalg.norm(c)
print(c.T @ Q(4) @ c)