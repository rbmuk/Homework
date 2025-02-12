import numpy as np

def f(x: float) -> float:
    return 2*(np.sqrt(x+1)-np.sqrt(x)) - (np.sqrt(x) - np.sqrt(x-2))

print(f(30))