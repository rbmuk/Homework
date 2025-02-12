import numpy as np
import itertools

sqrt2 = np.sqrt(2)

def Rx(theta: float):
    return np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
def Ry(gamma: float):
    return np.matrix([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
def Rz(alpha: float):
    return np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

R = 1/np.sqrt(3) * np.matrix([[1, 1, 1], [1, -1, 1], [-1, -1, 1]])
print(R)
Beta = 0
for x in itertools.product([-1, 1], repeat=3):
    Rx = R @ x
    print(Rx.T, "\n")
    Beta += np.max(np.absolute(Rx))
print(Beta/2**3)