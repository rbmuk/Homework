import numpy as np
import itertools as it
rng = np.random.default_rng()


def beta(A: np.ndarray) -> float:
    d = A.shape[1]
    combinations = np.array(list(it.product([-1,1], repeat=d)))
    return np.mean(np.linalg.norm(A @ combinations.T, axis=0, ord=np.inf))

def approx_beta(M: np.ndarray):
    X = rng.choice([-1,1], size=(len(M), 1000), p=[1/2, 1/2])
    return np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf))

def magic(n: int) -> np.ndarray:
    A = np.zeros((n,n))
    for i in range(n-1):
        A[i+1, i] = 1
    A[0, :] = 1
    return A

def Hs(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1., 1], [1, -1]])
    return np.kron(Hs(n-1), np.array([[1, 1], [1, -1]]))

def random(n: int) -> np.ndarray:
    return rng.choice([-1., 1], size=(2**n, 2**n))

def Xs(n: int) -> np.ndarray:
    return np.array(list(it.product([-1,1], repeat=2 ** n)))

H = Hs(8)
dist = (H ** 2).sum(axis=1).reshape(-1, 1) - 2 * H @ H + (H ** 2).sum(axis=1).reshape(1, -1)
print((2 ** 8) ** 2 - np.count_nonzero(dist) - 2 ** 8)