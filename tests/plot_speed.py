import numpy as np
import matplotlib.pyplot as plt

x = [.001, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
y_tree = np.array([85.05, 20.99, 5.23, 2.64, 1.397, .907, .584, .411, .311, .326, .873])
y_graph = np.array([59.8487, 33.5915, 13.7399, 8.16879, 6.09247, 3.98559, 3.03338, 2.49828, 2.17596, 1.91621, 2.13368])
y_bruteforce = np.array([26.9423, 26.9423, 26.9423, 26.9423, 26.9423, 26.9423, 26.9423, 26.9423, 26.9423, 26.9423, 26.9423])

y_tree_parallel = np.array([22.8952, 6.1502, 1.51648, 0.695098, 0.414156, 0.28228, 0.211139, 0.173647, 0.144507, 0.128288, 1.70808])
y_graph_parallel = np.array([22.2332, 14.7449, 4.51804, 2.77265, 2.1031, 1.95157, 1.40433, 1.38959, 1.32818, 1.13793, 2.36739])
y_bruteforce_parallel = np.array([12.7473, 12.7473, 12.7473, 12.7473, 12.7473, 12.7473, 12.7473, 12.7473, 12.7473, 12.7473, 12.7473])


plt.plot(x, y_tree / y_tree_parallel, label="tree method")
plt.plot(x, y_graph / y_graph_parallel, label="graph method")
plt.plot(x, y_bruteforce / y_bruteforce_parallel, label="bruteforce method")

plt.xlabel("Theta")

plt.ylabel("Speedup")
plt.title("Speedup vs Theta Threshold")
plt.legend()

# plt.show()
plt.savefig("speedup.png")