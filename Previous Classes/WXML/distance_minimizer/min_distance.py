import numpy as np

def loss(a: np.array, b: np.array, c: np.array) -> float:
    return (c-np.array([1,1,1])) @ (c - np.array([1,1,1])) + (c-np.array([1, -1,1])) @ (c - np.array([1, -1,1])) + (a-np.array([-1,1,1])) @ (a - np.array([-1,1,1])) + (b-np.array([-1, -1,1])) @ (b - np.array([-1, -1,1]))

def gradient_descent(a: np.array, b: np.array, c: np.array, learning_rate: float, num_iterations: int) -> np.array:
    for _ in range(num_iterations):
        gradient_a = 2 * (a - np.array([-1, 1,1]))
        gradient_b = 2 * (b - np.array([-1, -1,1]))
        gradient_c = 2 * (c - np.array([1, 1,1])) + 2 * (c - np.array([1, -1,1]))
        
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
        c -= learning_rate * gradient_c
    
    return np.array([a, b, c])

print(gradient_descent(np.array([0.,0.,0.]), np.array([0.,0.,0.]), np.array([0.,0.,0.]), 1e-1, 1000))