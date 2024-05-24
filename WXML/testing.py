import numpy as np
import itertools as it

def beta(A: np.ndarray) -> float:
    d = A.shape[1]
    combinations = np.array(list(it.product([-1,1], repeat=d)))
    return np.mean(np.linalg.norm(A @ combinations.T, axis=0, ord=np.inf))

# A = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
# A = [[1., 1., 1.],
#      [-1./3, -1./3, 1.]]
# A /= np.linalg.norm(A, axis=1, keepdims=True)
# print(f'inf-norm of A @ [-1,-1,1]: {np.linalg.norm(A @ np.array([1,-1,1]), ord=np.inf)}')
# print(f'beta(A): {beta(A)}')

# for i in range(1,15):
#     A = np.c_[np.c_[np.array(list(it.product([-1,1], repeat=i))), np.zeros((2**i, 2**i-i-1))], np.ones((2**i, 1))]
    # np.savetxt(f'matrices/{2**i}x{2**i}.csv', A, delimiter=',', fmt='%f')
#A = np.array([[1., -1, -1, 1,], [-1, 1, -1, 1], [1, 1, 1, 1], [-1, -1, 1, 1]])

def H(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1., 0], [1, 1]])
    return np.block([[H(n-1), H(n-1)], [H(n-1), -H(n-1)]])

H = H(8)
H /= np.linalg.norm(H, axis=1, ord=2)
print(beta(H))