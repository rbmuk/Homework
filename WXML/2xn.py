import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def construction(n: int):
    if n == 1:
        return np.array([[1., 1], [-1, -1]])
    M = construction(n - 1)
    M_dagger = np.copy(M)
    M_dagger[:,-1] = -M_dagger[:,-1]
    return np.block([[M, M_dagger], [M_dagger, M]])

n = 256

hist = []
ps = []

# for p in np.linspace(0, 0.9, 10):
#     ps.append(p)
#     p_hist = []
#     for i in tqdm(range(500)):
#         M = rng.choice([-1., 0., 1.], size=(n, n), p=[(1-p)/2, p, (1-p)/2])
#         M /= np.linalg.norm(M, axis=1, ord=2).reshape(-1, 1)

#         X = rng.choice([-1,1], size=(n, 1000), p=[1/2, 1/2])
#         p_hist.append(np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf)))
#     hist.append(np.mean(p_hist))
# print(np.max(hist))

# plt.plot(ps, hist)
# plt.xlabel('p')
# plt.ylabel('beta')
# plt.show()

def approx_beta(M: np.ndarray):
    X = rng.choice([-1,1], size=(len(M), 1000), p=[1/2, 1/2])
    return np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf))

M =construction(8)
M /= np.linalg.norm(M, axis=1, ord=2)
print(approx_beta(M))