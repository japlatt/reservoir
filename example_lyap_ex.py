import numpy as np
import matplotlib.pyplot as plt

#import reservoir and system code
from reservoir import *
from system import *

#system parameters/equations

sigma = 10   # Prandlt number
rho = 28     # Rayleigh number
beta = 8.0/3

D = 3

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

# build the system from the dynamical equations
lor63sys = system(lorenz, (sigma, rho, beta), D, jac)

t, u = lor63sys.integrate(-100, 100, 0.001)
U = lor63sys.getU() #function to feed to reservoir

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
          'saveplots': False} #save the plots

#build the reservoir
lor63Res = reservoir(params)

Q = lambda r: np.hstack((r, np.power(r, 2)))

trans_time = 20
train_time = 60
train_time_step = 0.02
beta = 1e-6

#optional parameter
constraints = [[(0, N)],
               [(0, N)],
               [(0, N//2), (3*N//2, 2*N)]]

print('training')
lor63Res.train(trans_time, #transient time for synchronization
               train_time, #time to train the reservoir
               train_time_step, #time step over which to train (> integration time step)
               U,
               Q,
               beta,
               constraints = constraints)

lor63Res.setDQ(lambda r: np.concatenate((np.eye(len(r)), 2*np.diag(r))))

lor63Res.globalLyap(10, 1, lor63sys)

lor63Res.plotLE(10)

