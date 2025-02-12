from numpy import random
import numpy as np

moneys = np.empty([100],dtype=int)
money = 100
for k in range(100):
    for n in range(1000):
        money -= 1
        m = random.randint(1, 3)
        if m == 2:
            money += 2
        moneys[k]=money
print(np.average(moneys))