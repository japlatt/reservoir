'''
This example calculates the variation in prediction for a particular
set of parameters.  Shows the random variation in prediction.

X gives the number of hosts

run: python3 -m charmrun.start ++numHosts X ++processPerHost 1 example_pred_variation.py ++nodelist nodelist.txt +isomalloc_sync
'''
from charm4py import charm
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import time
from functools import partial

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


def Q(r): return np.hstack((r, np.power(r, 2)))

def getPredData(time, res, acc):
    pred_acc = res.predict(time, acc = acc)
    x = res.predSysx0
    return pred_acc, x

def main(args):
    #system parameters/equations

    sigma = 10   # Prandlt number
    rho = 28     # Rayleigh number
    beta = 8.0/3

    D = 3

    # build the system from the dynamical equations
    lor63sys = syst.system(lorenz, (sigma, rho, beta), D, 0.001)

    #------------------------------------------------------------------------
    #build reservoir
    N = 2000
    sigma = 0.014

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
              'time_step': 0.001, #integration time step for reservoir
              'spinup_time': 20, #time it takes to synchronize res with sys
              'system': lor63sys, #system to run reservoir over
              'saveplots': False} #save the plots

    train_time = 60
    train_time_step = 0.02
    beta = 1e-6

    #optional parameter
    constraints = [[(0, N)],
                   [(0, N)],
                   [(0, N//2), (3*N//2, 2*N)]]


    num_res = 10 #build 10 reservoirs with the same parameters
    trials = 4000
    predictions = []
    X = []
    for i in range(num_res):
        startTrain = time()
        #build the reservoir
        print('initialize res')
        lor63Res = res.reservoir(params)

        lor63Res.train(train_time, #time to train the reservoir
                       train_time_step, #time step over which to train (> integration time step)
                       Q,
                       beta,
                       constraints = constraints)

        print(time() - startTrain)
        startTime = time()

        remote = partial(getPredData, res = lor63Res, acc = 3)

        prediction = np.array(charm.pool.map(remote, 15*np.ones(trials)))
        predictions.append(prediction[:, 0])
        X.append(prediction[:, 1])
        elapsed = time() - startTime
        print(elapsed)

    print('done')
    np.savez('pred_variation.npz', pred = predictions, x = X)
    exit()


charm.start(main)

