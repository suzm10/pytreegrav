import matplotlib.pyplot as plt

x = [0, .2, .4, .6, .8, 1]
y_tree = [94.765, 5.51529, 1.30163, 0.576726, 0.311827, 0.872802]
y_graph = [64.358, 13.8119, 5.29123, 2.70334, 1.79084, 1.92961]
y_bruteforce = [28.6415, 28.6415, 28.6415, 28.6415, 28.6415, 28.6415]

plt.plot(x, y_tree, label="tree method")
plt.plot(x, y_graph, label="graph method")
plt.plot(x, y_bruteforce, label="bruteforce method")

plt.xlabel("Theta")

plt.ylabel("Runtime")
plt.title("Runtime vs Theta Threshold")
plt.legend()

# plt.show()
plt.savefig("runtime.png")