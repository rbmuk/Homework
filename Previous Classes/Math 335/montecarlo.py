import numpy as np
from numpy import random

wins = np.empty(100, dtype=float)
for N in range(1, 100):
    winrate = 0
    for i in range(100):
        x = random.uniform(0, 1, [100])
        firstN = x[:N]
        maxfirst = firstN.max()
        strategy = -1
        for j in range(99-N):
            if x[N+j+1] >= maxfirst:
                strategy = x[N+j-1]
                break
        if (strategy == -1):
            strategy = x[99]
        totalmax = x.max()
        if (totalmax - strategy < 0.0000000001):
            winrate += 1
    wins[N] = winrate/1000

top5 = np.argpartition(wins, -5)[-5:]
print(np.round(sum(top5)/5))