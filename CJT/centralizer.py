from sympy import Matrix
from sympy import Matrix, symbols, Eq, solve, pprint

# Define the matrix
matrix = Matrix([[1, 2], [3, 4]])

def centralizer(matrix: Matrix):
    # Get the dimensions of the matrix
    rows, cols = matrix.shape

    # Define the variables
    variables = symbols(' '.join([f'a{i}{j}' for i in range(rows) for j in range(cols)]))

    # Define the general commuting matrix
    commuting_matrix = Matrix(variables).reshape(rows, cols)

    # Define the equation for matrix commutation
    commutation_eq = Eq(matrix * commuting_matrix, commuting_matrix * matrix)

    # Solve the equation to find the values of the variables
    solution = solve(commutation_eq, variables)

    # Convert the solution to a matrix
    solution_matrix = Matrix([[solution.get(var, var) for var in variables]]).reshape(rows, cols)

    # Print the explicit equations for the commuting matrices as a matrix
    return solution_matrix