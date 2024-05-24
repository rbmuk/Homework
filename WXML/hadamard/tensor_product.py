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

n = 5
H = Hs(n)
H /= np.sqrt(2**n)
print(approx_beta(H))
# for i in range(1, 10):
#     H = Hs(i)/np.sqrt(2**i)
#     r = random(i)/np.sqrt(2**i)
#     print(f'H beta value for shape {H.shape}: {approx_beta(H)}')
    # print(f'random beta value for shape {H.shape}: {approx_beta(r)}')