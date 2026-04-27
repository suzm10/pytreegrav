import numpy as np
from pytreegrav import Accel, Potential
from time import time

# num of particles
N = 10**5
# positions randomly sampled
x = np.random.rand(N, 3)
# masses - let the system have unit mass
m = np.repeat(1./N, N)
# softening radii - these are optional, assumed 0 if not provided
h = np.repeat(0.01, N)

# print(f"m: {m}")

# print(Accel(x, m, h))
# print(Potential(x, m, h))

t = time()
accel_tree = Accel(x, m, h, theta=0.4, method='tree')
print("Tree accel runtime: %gs"%(time() - t)); t = time()

accel_bruteforce = Accel(x, m, h, method='bruteforce')
print("Brute force accel runtime: %gs"%(time() - t)); t = time()

accel_hnsw = Accel(x, m, h, theta=0.4, method='hnsw')
print("HNSW accel runtime: %gs"%(time() - t)); t = time()

acc_error = np.sqrt(np.mean(np.sum((accel_hnsw-accel_bruteforce)**2,axis=1)))

print(f"acc_error hnsw: {acc_error}")

acc_error = np.sqrt(np.mean(np.sum((accel_tree-accel_bruteforce)**2,axis=1)))

print(f"acc_error tree: {acc_error}")

accel_tree = Accel(x, m, h, theta=0.4, method='tree', parallel=True)
print("Tree accel runtime in parallel: %gs"%(time() - t)); t = time()

accel_hnsw = Accel(x, m, h, theta=0.4, method='hnsw', parallel=True)
print("HNSW accel runtime in parallel: %gs"%(time() - t)); t = time()

# phi_tree = Potential(x, m, h, method='tree', parallel=True)
# print("Tree potential runtime: %gs"%(time() - t)); t = time()

# phit_bruteforce = Potential(x, m, h, method='bruteforce', parallel=True)
# print("Brute force potential runtime: %gs"%(time() - t)); t = time()

# thetas = .1, .2, .4, .8
# for theta in thetas:
#     t = time()
#     accel_tree = Accel(x, m, h, method='tree', theta=theta)
#     acc_error = np.sqrt(np.mean(np.sum((accel_tree-accel_bruteforce)**2,axis=1)))
#     print("theta=%g Runtime: %gs RMS force error: %g"%(theta, time()-t, acc_error))



def tree_results(theta):
    t = time()
    print(f"theta: {theta}")
    accel_tree = Accel(x, m, h, theta=theta, method='tree')
    print("Tree accel runtime: %gs"%(time() - t))

    acc_error = np.sqrt(np.mean(np.sum((accel_tree-accel_bruteforce)**2,axis=1)))
    print(f"acc_error tree: {acc_error}")

def graph_results(theta):
    t = time()
    print(f"theta: {theta}")
    accel_hnsw = Accel(x, m, h, theta=theta, method='hnsw')
    print("HNSW accel runtime: %gs"%(time() - t)); t = time()

    acc_error = np.sqrt(np.mean(np.sum((accel_hnsw-accel_bruteforce)**2,axis=1)))
    print(f"acc_error hnsw: {acc_error}")


# tree_results(.001)
# tree_results(.2)
# tree_results(.4)
# tree_results(.6)
# tree_results(.8)
# tree_results(1)

# graph_results(.001)
# graph_results(.2)
# graph_results(.4)
# graph_results(.6)
# graph_results(.8)
# graph_results(1)

