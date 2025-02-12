import numpy as np
from numpy.random import rand, randint, normal

eps = 0.01

def perfect_matching(n):
    vertices = [i for i in range(n)]
    A = np.zeros((n, n))
    while len(vertices) >= 2:
        vertex_1 = vertices[randint(0, len(vertices))]
        vertices.remove(vertex_1)
        vertex_2 = vertices[randint(0, len(vertices))]
        vertices.remove(vertex_2)
        A[vertex_1][vertex_2] = A[vertex_2][vertex_1] = 1
    return A

def d_regular(d, n):
    A = np.zeros((n, n))
    for i in range(d):
        A += perfect_matching(n)
    return A

def second_largest_eigenvalue(A):
    n = len(A[0])
    B = A @ A
    v1 = np.array([1/np.sqrt(n) for i in range(n)])
    v2 = normal(0, 1, n)
    v2 = v2 - (v1 @ v2) * v1
    for i in range(int(np.log2(n)/eps)):
        v2 = B @ v2
        v2 = v2 - (v1 @ v2) * v1
        v2 = v2 / np.linalg.norm(v2)
    return np.sqrt(v2 @ (B @ v2))


for d in range(3, 11):
    A = d_regular(d, 10000)
    print("d = ", d, ", n = 10000")
    print("Approximate second largest eigenvalue: ", second_largest_eigenvalue(A/d))
    print("True second largest eigenvalue: ", sorted(np.abs(np.linalg.eigvals(A/d)))[-2])
