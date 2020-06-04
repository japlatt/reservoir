import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from progress.bar import Bar
import gc

'''
Compute the global lyapunov exponents of a dynamical system
defined by f with jacobian fjac.

f: dynamical system equations f(x, t, p)
fjac: jacobian of dynamical system f(x, t, p)
t: array, time over which to run QR decomposition algorithm
dt: integration dt for the dynamical system f
pf: parameters for f
pjac: parameters jacobian
x0: start point for computing LE
D: dimension of the system
'''
def computeLE(f, fjac, t, dt, pf, pjac, x0, D):

    def dPhi_dt(Phi, t, y, p, Dim):
        """ The variational equation """
        rPhi = np.reshape(Phi, (Dim, Dim))
        rdPhi = np.dot(fjac(y, t, p), rPhi)
        return rdPhi.flatten()

    def dSdt(t, S, pf, pjac, Dim):
        """
        Differential equations for combined state/variational matrix
        propagation. This combined state is called S.
        """
        y = S[:Dim]
        Phi = S[Dim:]
        return np.append(f(y, t, pf), dPhi_dt(Phi, t, y, pjac, Dim))

    LE = np.zeros((len(t)-1, D))
    Phi0 = np.eye(D).flatten()
    Ssol = np.append(x0, Phi0)
    print("Integrating system for LE calculation...")

    with Bar('Processing', max=len(t)) as bar:
        for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
            sol = odeint(dSdt,
                         Ssol,
                         np.arange(t1, t2, dt),
                         args = (pf, pjac, D),
                         tfirst = True)
            x = sol[-1][:D]
            rPhi = sol[-1][D:].reshape(D, D)
            
            # perform QR decomposition on Phi
            Q,R = np.linalg.qr(rPhi)
            Ssol = np.append(x, Q.flatten())
            LE[i] = np.abs(np.diag(R))
            bar.next()
            gc.collect()


    # compute LEs
    print("Computing LE spectrum...")
    LE = np.cumsum(np.log(LE),axis=0) / np.tile(t[1:],(D,1)).T
    print('done')
    return LE

'''
plot num_exp number of exponents
'''
def plotNLyapExp(LE, t, num_exp, fig, ax):
    fig.set_tight_layout(True) 

    colors = ['dodgerblue']*num_exp 
    for i in range(num_exp): 
        ax.plot(t[1:], LE[:,i], color=colors[i%4], label=r'$\lambda_{%d}$'%(i+1,), lw=1.2)
    ax.set_ylabel(r'$\lambda$', size=14) 
    ax.set_xlabel(r'$t$', size=14) 
    ax.set_title('%d largest Lyapunov exponents'%(num_exp,)) 


'''
Kaplan-Yorke dimension of the system
'''
def KY_dim(LE):
    LE = LE[-1]
    j = -1
    lyap_sum = 0
    for l in LE:
        j+=1
        temp = lyap_sum+l
        if temp < 0: break
        lyap_sum = temp
    return j+lyap_sum/abs(LE[j])