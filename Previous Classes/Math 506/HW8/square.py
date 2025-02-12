import turtle

# Create a turtle object
t = turtle.Turtle()

# Define a list of colors
colors = ["red", "green", "blue", "yellow"]

# Loop through the colors and draw the square
for color in colors:
    t.fillcolor(color)
    t.begin_fill()
    for _ in range(4):
        t.forward(100)
        t.right(90)
    t.end_fill()

# Hide the turtle
t.hideturtle()

# Exit on click
turtle.exitonclick()