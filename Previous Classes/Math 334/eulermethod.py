import math

stepsize = float(input("stepsize: "))
dydt = input("dy/dt: ")
initial_point = input("initial point (no parenthesis): ")
number_of_steps = input("number_of_times: ")
initial_point = initial_point.split(", ")
y = float(initial_point[1])
t = float(initial_point[0])

def f():
    global t
    global y
    y = y + stepsize * eval(dydt)
    t += stepsize
    print("new point: (" + str(t) + ", " + str(y) + ")")

for i in range(int(number_of_steps)):
    f()
