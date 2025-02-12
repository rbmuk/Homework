import numpy as np
import pandas as pd

def knapsack(value: np.array, weight: np.array, W: int) -> np.ndarray:
    n = len(weight)
    M = np.zeros((n, W+1), dtype='int')
    for w in range(W+1):
        M[0, w] = 0
    for i in range(0,n):
        for w in range(1, W+1):
            if (weight[i] > w and i != 0):
                M[i,w] = M[i-1, w]
            else:
                M[i, w] = max(M[i-1, w], value[i] + M[i-1, w-weight[i]])
    return M

W = 14
value = [1, 2, 4, 5, 7]
weight = [1,3 , 5, 7, 9]
column_names = list(range(W+1))
row_names = list(range(1, len(value)+1))
df = pd.DataFrame(knapsack(value, weight, W), index=row_names, columns=column_names)
df.to_csv('knapsack.csv', index=True, header=True, sep = ',')