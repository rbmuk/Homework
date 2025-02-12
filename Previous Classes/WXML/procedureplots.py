from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

# Define the vertices of the cube
vertices = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
]

# Connect the vertices with edges
edges = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
]

# Create a new figure and set up a 3D plot
fig = plt.figure()

def plot(vertices_to_plot: list, color: str, i):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    for vertex in vertices_to_plot:
        ax.scatter(*vertex, color=color, s=50)

    for edge in edges:
        ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]], color='black')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{len(vertices_to_plot)}x{4}')

plot(vertices, 'red', 1)

vertices_2 = [
    (-1, -1, 0), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
]
plot(vertices_2, 'red', 2)

vertices_3 = [
    (-1, -1, 0), (-1, 1, 0),
    (1, -1, 0), (1, 1, 0)
]
plot(vertices_3, 'red', 3)



# Show the plot
plt.show()