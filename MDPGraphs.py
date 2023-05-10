
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

area = [25,28,100]
valTime = [0.011,0.011,0.086]
policyTime = [0.002,0.002,0.022]

valSpace = [3877574,3877874,3888822]
policySpace = [3877518,3877798,3891166]

plt.plot(area, valTime, c='teal', label='Value Iteration')
plt.plot(area, policyTime, c='darkorange', label='Policty Iteration')

plt.xlabel("Grid Size (Area)")
plt.ylabel('Time in seconds')
plt.title("MDP Algorithms time taken for different maze sizes")
plt.legend(loc="upper right")
plt.show()

plt.plot(area, valSpace, c='teal', label='Value Iteration')
plt.plot(area, policySpace, c='darkorange', label='Policy Iteration')

plt.xlabel("Grid Size (Area)")
plt.ylabel('Memory Usage')
plt.title("MDP Algorithms memory usage for different maze sizes")
plt.legend(loc="upper right")
plt.show()