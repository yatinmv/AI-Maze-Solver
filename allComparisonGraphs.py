
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

area = [25,28,100]
valTime = [0.011,0.011,0.086]
policyTime = [0.002,0.002,0.022]
dfsTime = [0.00005,0.00004,0.00007]
bfsTime = [0.0002,0.0002,0.0004]
aStarTime = [0.0003,0.0003,0.0012]

valSpace = [3877574,3877874,3888822]
policySpace = [3877518,3877798,3891166]
dfsSpace = [3877558,3877726,3891206]
bfsSpace = [3877446,3877726,3891094]
aStarSpace = [3877446,3877726,3891094]


plt.plot(area, valTime, c='teal', label='Value Iteration')
plt.plot(area, policyTime, c='darkorange', label='Policty Iteration')
plt.plot(area, dfsTime, c='green', label='DFS')
plt.plot(area, bfsTime, c='red', label='BFS')
plt.plot(area, aStarTime, c='purple', label='A Star')

plt.xlabel("Grid Size (Area)")
plt.ylabel('Time in seconds')
plt.title("Time taken by all algorithms for different maze sizes")
plt.legend(loc="upper right")
plt.show()

plt.plot(area, valSpace, c='teal', label='Value Iteration')
plt.plot(area, policySpace, c='darkorange', label='Policy Iteration')
plt.plot(area, dfsSpace, c='green', label='DFS')
plt.plot(area, bfsSpace, c='red', label='BFS')
plt.plot(area, aStarSpace, c='purple', label='A Star')

plt.xlabel("Grid Size (Area)")
plt.ylabel('Memory Usage')
plt.title("Memory usage of all algorithms for different maze sizes")
plt.legend(loc="upper right")
plt.show()