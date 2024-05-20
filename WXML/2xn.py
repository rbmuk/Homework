import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
rng = np.random.default_rng()

n = 512

def dist(M: np.ndarray, N: np.ndarray):
    xs = np.sum(M**2, axis=1).reshape(-1, 1)
    ys = np.sum(N**2, axis=1).reshape(1, -1)
    return np.mean(-2 * M @ N.T + xs + ys)

def approx_beta(M: np.ndarray):
    X = rng.choice([-1,1], size=(len(M), 1000), p=[1/2, 1/2])
    return np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf))

hist = []
ps = []
Ms = []

p = 0
# for p in np.linspace(0, 0.9, 10):
#ps.append(p)
#p_hist = []
for i in tqdm(range(500)):
    M = rng.choice([-1., 0., 1.], size=(n, n), p=[(1-p)/2, p, (1-p)/2])
    M /= np.linalg.norm(M, axis=1, ord=2).reshape(-1, 1)

    X = rng.choice([-1,1], size=(n, 1000), p=[1/2, 1/2])
    Ms.append(M)
    hist.append(np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf)))

max = np.argmax(hist)
min = np.argmin(hist)
print(hist[max], hist[min])
print('max average hamming distance: ', dist(Ms[max], Ms[max]))
print('min average hamming distance: ', dist(Ms[min], Ms[min]))
    #p_hist.append(np.mean(np.linalg.norm(M @ X, axis=0, ord=np.inf)))
#hist.append(np.mean(p_hist))
# print(np.max(hist))

# plt.plot(ps, hist)
# plt.xlabel('p')
# plt.ylabel('beta')
# plt.show()
