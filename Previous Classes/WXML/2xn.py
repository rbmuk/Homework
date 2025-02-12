import numpy as np
from tqdm import tqdm
import itertools as it
rng = np.random.default_rng()

n = 256

def dist(M: np.ndarray, N: np.ndarray):
    xs = np.sum(M**2, axis=1).reshape(-1, 1)
    ys = np.sum(N**2, axis=1).reshape(1, -1)
    return np.mean(-2 * M @ N.T + xs + ys)

def approx_beta(M: np.ndarray):
    X = rng.choice([-1,1], size=(M.shape[1], 1000), p=[1/2, 1/2])
    return np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf))

def H(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1., 0], [1, 1]])
    return np.block([[H(n-1), H(n-1)], [H(n-1), -H(n-1)]])

def beta(A: np.ndarray) -> float:
    d = A.shape[1]
    combinations = np.array(list(it.product([-1,1], repeat=d)))
    return np.mean(np.linalg.norm(A @ combinations.T, axis=0, ord=np.inf))



betas = []
for i in tqdm(range(1000)):
    M = rng.choice([-1., 1], size=(2, 256), p=[0.5, 0.5])
    M /= np.linalg.norm(M, axis=1, ord=2).reshape(-1, 1)
    betas.append(approx_beta(M))
print(np.max(betas))

# H = H(5)
# print(H.shape)
# H /= np.linalg.norm(H, axis=1, ord=2)
# print(H)
# print(f'approx: {approx_beta(H)}')
# print(f'true: {beta(H)}')

# hist = []
# ps = []

# for p in np.linspace(0, 0.9, 10):
#     ps.append(p)
#     p_hist = []
#     for i in tqdm(range(2000)):
#         M = rng.choice([-1., 0., 1.], size=(n, n), p=[(1-p)/2, p, (1-p)/2])
#         M /= np.linalg.norm(M, axis=1, ord=2).reshape(-1, 1)

#         X = rng.choice([-1,1], size=(n, 1000), p=[1/2, 1/2])
#         p_hist.append(np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf)))
#     hist.append(np.max(p_hist))

# plt.plot(ps, hist)
# plt.xlabel('p')
# plt.ylabel('beta')
# plt.show()