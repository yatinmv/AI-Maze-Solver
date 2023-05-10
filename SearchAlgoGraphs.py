
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

area = [25,28,100]
dfs = [0.017,0.015,0.031]
bfs = [0.031,0.036,0.275]
astar = [0.058,0.066,0.195]
dfsSpace = [16,14,22]
bfsSpace = [25,28,100]
aStarSpace = [11,12,28]

plt.plot(area, bfs, c='teal', label='BFS')
plt.plot(area, dfs, c='darkorange', label='DFS')
plt.plot(area, astar, c='red', label='A*')


plt.xlabel("Grid Size (Area)")
plt.ylabel('Time in seconds')
plt.title("Time taken by different search algoritms for different maze sizes")
plt.legend(loc="upper right")
plt.show()

plt.plot(area, bfsSpace, c='teal', label='BFS')
plt.plot(area, dfsSpace, c='darkorange', label='DFS')
plt.plot(area, aStarSpace, c='red', label='A*')


plt.xlabel("Grid Size (Area)")
plt.ylabel('Search Space')
plt.title("Search Space covered by different search algoritms for different maze sizes")
plt.legend(loc="upper right")
plt.show()