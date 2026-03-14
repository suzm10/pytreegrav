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

# print(Accel(x, m, h))
# print(Potential(x, m, h))

t = time()
# accel_tree = Accel(x, m, h, method='tree', parallel=True)
# print("Tree accel runtime: %gs"%(time() - t)); t = time()

# accel_bruteforce = Accel(x, m, h, method='bruteforce')
# print("Brute force accel runtime: %gs"%(time() - t)); t = time()

accel_hnsw = Accel(x, m, h, method='hnsw')

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