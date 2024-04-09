import numpy as np
A = 1./2 * np.matrix([[-1, -1, np.sqrt(2)], 
               [-np.sqrt(2), 0, np.sqrt(2)], 
               [1, 1, np.sqrt(2)]])
Q = np.matrix([[1/2, 1/2, 1/np.sqrt(2)], 
              [-1/2, -1/2, 1/np.sqrt(2)],
              [1/np.sqrt(2), 1/np.sqrt(2), 0]])
# Calculate the coordinates of the hypercube vertices after applying matrix A
hypercube_vertices = np.array([[-1, -1, -1],
                                [-1, -1, 1],
                                [-1, 1, -1],
                                [-1, 1, 1],
                                [1, -1, -1],
                                [1, -1, 1],
                                [1, 1, -1],
                                [1, 1, 1]]).T

transformed_vertices = np.dot(Q, hypercube_vertices)

# Calculate the average maximum coordinate
average_max_coordinate = np.mean(np.max(np.abs(transformed_vertices), axis=0))

print(average_max_coordinate)