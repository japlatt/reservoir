'''
This example calculates the lyapunov exponents for the predicting
reservoir.  The algorithm takes a long time and a significant
amount of memory.

python3 -m charmrun.start ++numHosts X ++processPerHost 1 example_lyap_ex.py ++nodelist nodelist.txt +isomalloc_sync
'''
import numpy as np
import matplotlib.pyplot as plt
import dill
import sys
from time import time

sys.path.insert(1, '../')
#import reservoir and system code
import reservoir as res
import system as syst

#lorenz63 model
def lorenz(n, t, p):
    sigma, rho, beta = p
    x, y, z = n
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y - beta*z
    dXdt = [dxdt, dydt, dzdt]
    return dXdt

def jac(X, t, p):
    sigma, rho, beta = p
    x, y, z = X
    return np.array([[-sigma, sigma, 0],
                     [-z+rho, -1, -x],
                     [y, x, -beta]])

def Q(r): return np.hstack((r, np.power(r, 2)))

def dQ(r): return np.concatenate((np.eye(len(r), dtype = np.float32), 2*np.diag(r)))

if __name__ == '__main__':

    #system parameters/equations

    sigma = 10   # Prandlt number
    rho = 28     # Rayleigh number
    beta = 8.0/3

    D = 3

    # build the system from the dynamical equations
    lor63sys = syst.system(lorenz, (sigma, rho, beta), D, 0.001, fjac = jac)


    #build reservoir
    N = 2000
    sigma = 0.014

    res_time_step = 0.001
    # dr/dt = gamma * (-r + tanh(M*r+Win*u))
    params = {'name': 'Lor63_N2000',
              'N': N, #number of neurons
              'D': D, #dimension of the input system
              'gamma': 10, #time constant for the system
              'M_a': -1, #M connectivity matrix lower bound
              'M_b': 1, #M connectivity matrix upper bound
              'M_pnz': 0.02, #M connectivity matrix sparsity
              'M_SR': 0.9, #M connectivity matrix spectral radius
              'Win_a': -sigma, #Win input matrix lower bound
              'Win_b': sigma, #Win input matrix upper bound
              'time_step': res_time_step, #integration time step for reservoir
              'spinup_time': 40, #time it takes to synchronize res with sys
              'system': lor63sys, #system to run reservoir over
              'saveplots': False} #save the plots

    #build the reservoir
    lor63Res = res.reservoir(params)

    train_time = 60
    train_time_step = 0.02
    beta = 1e-6

    #optional parameter
    constraints = [[(0, N)],
                   [(0, N)],
                   [(0, N//2), (3*N//2, 2*N)]]

    print('training')
    startTime = time()
    lor63Res.train(train_time, #time to train the reservoir
                   train_time_step, #time step over which to train (> integration time step)
                   Q,
                   beta,
                   constraints = constraints)
    print('finished training in: ', time() - startTime)

    lor63Res.setDQ(dQ)

    print('computing lyapunov spectrum')
    startTime = time()
    lor63Res.globalLyap_TLM(25, 0.01)
    print('computed exponents in: ', time() - startTime)
    np.savez('LE.npz', LE = lor63Res.LE, t = lor63Res.LE_t)

