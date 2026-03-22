from numpy import sqrt, empty, zeros, empty_like, zeros_like, dot, fabs
from numba import njit, prange, get_num_threads, set_parallel_chunksize, int64, float64
from math import copysign
from .kernel import *
from .misc import *
import numpy as np
from scipy.spatial.transform import Rotation as R

@njit(fastmath=True)
def AccelWalk_HNSW(pos, target_idx, graph, softening=0, theta=0.7, G=1.0):
    accel = np.zeros(3, dtype=np.float64)

    stack = np.full(graph.MaxNumLevels * graph.M, -1, dtype=int64)
    stack[0] = graph.EntryPoint
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        if node_idx == -1:
            continue
        
        # calculate dist to post
        dx = graph.Coordinates[node_idx, 0] - pos[0]
        dy = graph.Coordinates[node_idx, 1] - pos[1]
        dz = graph.Coordinates[node_idx, 2] - pos[2]
        r2 = dx*dx + dy*dy + dz*dz + graph.Softenings[node_idx]**2
        r = np.sqrt(r2)

        # node is a leaf node
        if node_idx < graph.N:
            if node_idx != target_idx:
                fac = (graph.Masses[node_idx]) / (r2 * r)
                accel[0] += fac * dx
                accel[1] += fac * dy
                accel[2] += fac * dz
        else:
            # node is sufficiently far away
            if theta > (graph.Radii[node_idx] / r):
                fac = (graph.Masses[node_idx]) / (r2 * r)
                accel[0] += fac * dx
                accel[1] += fac * dy
                accel[2] += fac * dz
            else:
                for i in range(graph.M):
                    child_idx = graph.DownwardLinks[node_idx, i]
                    if child_idx != -1:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1

    return accel

def AccelTarget_graph(pos_target, softening_target, graph, theta=0.7, G=1.0, quadrupole=False):
    """Returns the gravitational acceleration at the specified points, given an HNSW graph containing the mass distribution
    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the field
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all accel computations
    graph -- Graph instance containing the positions, masses, and softenings of the source particles
    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)
    Returns:
    shape (N,3) array of acceleration values at each point in pos_target
    """
    # if softening_target is None:
    #     softening_target = zeros(pos_target.shape[0])
    result = empty(pos_target.shape)
    # set_parallel_chunksize(10000)

    N = len(pos_target)

    for i in prange(N):
        result[i] = G * AccelWalk_HNSW(pos_target[i], i, graph, softening_target[i], theta, G)
    
    return result


AccelTarget_graph = njit(AccelTarget_graph, fastmath=True)