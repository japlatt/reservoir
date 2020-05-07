'''
Helper functions for reservoir_ott18.py
'''
import numpy as np
from scipy.integrate import odeint
import scipy as sp
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.stats import uniform 
import scipy.linalg.blas


#predicting reservoir
def predicting(r, t, gamma, sig, M, Win, Wout):
    u_pred = np.dot(Wout, Q(r).T)
    MR = np.dot(M, r)
    WU = np.dot(Win, u_pred)
    drdt = gamma * (-r + np.tanh(MR+sig * WU))
    return drdt

#lorenz63 model
def lorenz(n, t, sigma, rho, beta):
    x, y, z = n
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y - beta*z
    return dxdt, dydt, dzdt

#listening reservoir
def listening(r, t, U, gamma, sig, M, Win):
    MR = np.dot(M, r)
    WU = np.dot(Win, U(t))
    drdt = gamma * (-r + np.tanh(MR+sig*WU))
    return drdt

#get some input data
def getInputData(system, u0, dt, start, time):
    if system == 'lorenz63':
        # Lorenz system
        sigma = 10   # Prandlt number
        rho = 28     # Rayleigh number
        beta = 8.0/3

        t = sp.arange(start, start+time, dt)
        u = odeint(lorenz, u0, t, args = (sigma, rho, beta))

        D = u.shape[1] #dimensionality of system

        u_arr = []

        for i in range(D):
            u_arr.append(interp1d(t, u[:, i], fill_value="extrapolate"))

        U = lambda t: np.array([x(t) for x in u_arr])
    return U, u, D, t

'''
Get the connection matrix for the reservoir

N: Number of Neurons
pnz: probability of element being nonzero
SR: spectral radius or magnitude of principle eigenvalue
'''

def getConnectionMat(N, pnz = 0.02, SR = 0.9):
    #set elements in M from U(a, b) and enforce sparsity
    a = -1
    b = 1
    M = sparse.random(N, N, density = pnz, data_rvs = uniform(loc=a, scale=(b-a)).rvs)
    M = sparse.csr_matrix(M)
    
    #rescale to specified spectral radius
    SRM = np.abs(sparse.linalg.eigs(M, k = 1, which='LM', return_eigenvectors=False))
    M = M.multiply(SR/SRM)
    return np.squeeze(np.asarray(M.todense(), order = 'C'))

'''
Get the input matrix for the reservoir

N: number of neurons
D: dimensionality of the input
'''
def getInputMat(N, D):
    R = np.argmax(np.random.rand(N, D), axis = 1)
    mask = np.zeros((N, D))
    for i, j in enumerate(R):
        mask[i][j] = 1
    Win = np.multiply(np.random.uniform(-1, 1, (N, D)), mask)
    return np.array(Win, order = 'C')


'''Takes in the optimization parameters and puts them into the
structure of Wout.  This basically adds constraints on Wout.'''
# def makeWout(params, N, Wout):
#     Wout[0, :N] = params[:N]
#     Wout[1, :N] = params[N:2*N]
#     Wout[2, :N//2] = params[2*N:5*N//2]
#     Wout[2, 3*N//2:] = params[5*N//2:]

#cost function to optimize over.  Least squares with Tikenov regularization
def cost(q, u, beta, Wout):
    # makeWout(params, N, Wout)
    z = scipy.linalg.blas.dgemm(alpha=1.0, a=Wout, b=q)
    cst = np.linalg.norm(np.subtract(z,u))**2
    cst += beta*np.linalg.norm(Wout)**2
    return cst


def Q(r): 
    r2 = np.power(r, 2)
    q = np.hstack((r, r2))
    return q