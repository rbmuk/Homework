from sympy import Matrix, symbols, diag, pprint, init_printing
import itertools
from centralizer import centralizer
from tqdm import tqdm

init_printing(wrap_line=False)

def find_minors(matrix: Matrix, size: int):
    rows, cols = matrix.shape
    if size > rows or size > cols:
        raise ValueError("Size is larger than matrix dimensions")

    rows_combs = itertools.combinations(range(rows), size)
    
    minors = []
    for rows_comb in rows_combs:
        cols_combs = itertools.combinations(range(cols), size)
        for cols_comb in tqdm(cols_combs):
            minor = matrix[rows_comb, cols_comb]
            minors.append(minor.det())

    return minors

# Example usage
x = diag(Matrix.jordan_block(3, 0), Matrix.jordan_block(2, 0), Matrix.jordan_block(5, 0), Matrix.jordan_block(5, 0), Matrix.jordan_block(5, 0), Matrix.jordan_block(5, 0), Matrix.jordan_block(5, 0))
y = diag(Matrix.jordan_block(3, 1), Matrix.jordan_block(2, 1), Matrix.jordan_block(5, 3), Matrix.jordan_block(5, 3), Matrix.jordan_block(5, 3), Matrix.jordan_block(5, 3), Matrix.jordan_block(5, 3))

u = symbols('u')
A = x + u * y
print((x ** 4).rank())
pprint(A ** 4)
pprint((A**4).rank())