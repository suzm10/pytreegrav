import matplotlib.pyplot as plt

"""
Tree accel runtime: 9.16039s
Brute force accel runtime: 26.9423s
HNSW accel runtime: 9.21783s
acc_error hnsw: 0.0031988282139724395
acc_error tree: 0.001518533667780026
Tree accel runtime in parallel: 2.5829s
HNSW accel runtime in parallel: 2.92726s
theta: 0.001
Tree accel runtime: 85.0463s
acc_error tree: 7.154056217625923e-13
theta: 0.1
Tree accel runtime: 20.9927s
acc_error tree: 3.9730703552061927e-05
theta: 0.2
Tree accel runtime: 5.23077s
acc_error tree: 0.0002540743816587267
theta: 0.3
Tree accel runtime: 2.63989s
acc_error tree: 0.000728923221160539
theta: 0.4
Tree accel runtime: 1.39731s
acc_error tree: 0.001518533667780026
theta: 0.5
Tree accel runtime: 0.906902s
acc_error tree: 0.002668556941669399
theta: 0.6
Tree accel runtime: 0.584035s
acc_error tree: 0.004277667438164125
theta: 0.7
Tree accel runtime: 0.410843s
acc_error tree: 0.006858386607810903
theta: 0.8
Tree accel runtime: 0.310724s
acc_error tree: 0.01072376825923052
theta: 0.9
Tree accel runtime: 0.326237s
acc_error tree: 0.01680156772676213
theta: 1
Tree accel runtime: 0.873065s
acc_error tree: 0.026413094463589536
theta: 0.001
HNSW accel runtime: 59.8487s
acc_error hnsw: 2.21951651390986e-14
theta: 0.1
HNSW accel runtime: 33.5915s
acc_error hnsw: 0.0001571065526670659
theta: 0.2
HNSW accel runtime: 13.7399s
acc_error hnsw: 0.0006307317591696173
theta: 0.3
HNSW accel runtime: 8.16879s
acc_error hnsw: 0.0015542509102941602
theta: 0.4
HNSW accel runtime: 6.09247s
acc_error hnsw: 0.0031988282139724395
theta: 0.5
HNSW accel runtime: 3.98559s
acc_error hnsw: 0.0056005649164569595
theta: 0.6
HNSW accel runtime: 3.03338s
acc_error hnsw: 0.009408435368332125
theta: 0.7
HNSW accel runtime: 2.49828s
acc_error hnsw: 0.015412960229885217
theta: 0.8
HNSW accel runtime: 2.17596s
acc_error hnsw: 0.024081708687409564
theta: 0.9
HNSW accel runtime: 1.91621s
acc_error hnsw: 0.0367584156876394
theta: 1
HNSW accel runtime: 2.13368s
acc_error hnsw: 0.054415223034618175
"""

# x = [.001, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
# y_tree = [0, 0.00003973*100, 0.000254*100, 0.00073*100, 0.00152*100, 0.002669*100, 0.004278*100, 0.006858*100, 0.0107*100, 0.0168*100, 0.0264*100]
# y_graph = [0, 0.0001571*100, 0.0006307*100, 0.00155*100, 0.0031988*100, 0.0056*100, 0.0094*100, 0.0154*100, 0.0241*100, 0.0368*100, 0.0544*100]
# y_bruteforce = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


x = [.001, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
y_tree = [0, 0.00003973*100, 0.000254*100, 0.00073*100, 0.00152*100, 0.002669*100, 0.004278*100, 0.006858*100, 0.0107*100, 0.0168*100, 0.0264*100]
y_graph = [0, 0.0001571*100, 0.0006307*100, 0.00155*100, 0.0031988*100, 0.0056*100, 0.0094*100, 0.0154*100, 0.0241*100, 0.0368*100, 0.0544*100]
y_bruteforce = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

plt.plot(x, y_tree, label="tree method")
plt.plot(x, y_graph, label="graph method")
plt.plot(x, y_bruteforce, label="bruteforce method")

plt.xlabel("Theta")

plt.ylabel("Percent Inaccuracy")
plt.title("Inaccuracy vs Theta Threshold")
plt.legend()

# plt.show()
plt.savefig("accuracy.png")