import matplotlib.pyplot as plt

x = [0, .2, .4, .6, .8, 1]
y_tree = [0, 0.00025373429450084616*100, 0.0015382747435710013*100, 0.004283851093261351*100, 0.010549867943909772*100, 0.02599165663637972*100]
y_graph = [0.09992300503800987*100, 0.09987858905672971*100, 0.09990718326977983*100, 0.10107907454802308*100, 0.10836579945338987*100, 0.1706942424242548*100]
y_bruteforce = [0, 0, 0, 0, 0, 0]

plt.plot(x, y_tree, label="tree method")
plt.plot(x, y_graph, label="graph method")
plt.plot(x, y_bruteforce, label="bruteforce method")

plt.xlabel("Theta")

plt.ylabel("Percent Inaccuracy")
plt.title("Inaccuracy vs Theta Threshold")
plt.legend()

# plt.show()
plt.savefig("accuracy.png")