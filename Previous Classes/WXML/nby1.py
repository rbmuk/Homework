import numpy as np
import itertools

from matplotlib import pyplot as plt

def beta(x: np.ndarray) -> float:
    total = 0
    for y in itertools.product([-1, 1], repeat=x.shape[0]):
        total += np.abs(x @ np.array(y))
    return total / 2**x.shape[0]

def f(x: float) -> float:
    return x**2 * np.power(0.91, x)

x = np.linspace(0, 50, 100)

plt.subplot(1, 2, 1)
plt.plot(x, f(x), color='green')
max_x = x[np.argmax(f(x))]
max_y = f(max_x)
plt.annotate(f'Max: x={max_x:.2f}', (max_x, max_y), xytext=(max_x+0.5, max_y+0.5), color='blue')
plt.plot(max_x, max_y, marker='o', markersize=8, color='blue')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x)')

plt.subplot(1, 2, 2)
x = range(2, 23)
y = [1/np.sqrt(i) * beta(np.ones(i)) for i in x]
plt.plot(x, y, color='red')
plt.xlabel('dimension')
plt.ylabel('Beta')
plt.title('Plot of Beta')
plt.show()
