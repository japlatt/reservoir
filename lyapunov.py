import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from progress.bar import Bar
from scipy.interpolate import interp1d
import gc



########### GLOBAL EXPONENTS ##############################

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

    with Bar('Processing', max=len(t)-1) as bar:
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
        ax.plot(t, LE[:,i], color=colors[i%4], label=r'$\lambda_{%d}$'%(i+1,), lw=1.2)
    ax.set_ylabel(r'$\lambda$', size=14) 
    ax.set_xlabel(r'$t$', size=14) 
    ax.set_title('%d largest Lyapunov exponents'%(num_exp,)) 


'''
Kaplan-Yorke dimension of the system


'''
def KY_dim(LE):
    lyap_sum = 0
    for j, l in enumerate(LE):
        temp = lyap_sum+l
        if temp < 0: break
        lyap_sum = temp
    return j+lyap_sum/abs(LE[j])



###### GLOBAL EXP TLM #######################

def computeLE_TLM(x, t, fjac, pjac, name, rescale_interval = 10, num_save = -1, savetxt = False):
    maxit, D = np.shape(x)
    I = np.eye(D, dtype = np.float32)
    M2 = np.eye(D, dtype = np.float32)

    if num_save < 0: num_save = D
    if savetxt: f = open(name+'_LE.txt', 'bw+')

    LE = np.zeros((int(maxit/rescale_interval), D), dtype = np.float32)
    l = 0
    prev_R = np.zeros(D)
    with Bar('Processing', max=maxit/rescale_interval) as bar:
        for i in range(maxit):
            dt = t[i+1] - t[i] if (i < maxit-1) else t[-1] - t[-2]

            # Evaluate Jacobian
            Df = fjac(x[i], t[i], pjac)

            # Compute approximate linear propagator
            M = I + Df*dt
            M2 = np.dot(M, M2)


            # Rescale via QR decomposition
            if (np.mod(i+1,rescale_interval)==0):
                Q,R = np.linalg.qr(M2)

                curr_R = np.log(np.abs(np.diag(R)))
                LE[l] = (prev_R+curr_R)/t[i]
                prev_R += curr_R

                #save concurrently
                if savetxt:
                    np.savetxt(f, [LE[l][:num_save]])
                    f.flush()

                M2 = Q #rescale M2
                l+=1
                bar.next()
    if savetxt > 0: f.close()
    return LE[:, :num_save], KY_dim(LE[-1])


###### LOCAL EXPONENTS ###########################

'''
Perform a recursive QR decomposition on a matrix

A: [A(L), A(L-1),...,A(1)] in order to find the
eigenvalues of A in a numerically stable way in the 
case that A is ill-conditioned.

T: time over which to compute the LLE

num_it: the number of times to perform recursion.


returns:
lRQR = lyapunov exponents from max to min
Q = need to check if this is close to identity matrix.
Up to a minus sign. abs(Q) should be arbitrarily close to the identity matrix.
'''
def rec_QR(A, T, num_it = 3):
    M_arr = A
    Q0 = np.eye(np.shape(A)[1], dtype=np.float32)
    for _ in range(num_it):
        L, N, _ = np.shape(M_arr)
        Q_prev = Q0
        R_arr = np.zeros((L, N, N), dtype=np.float32)
        for i in range(L-1, -1, -1):
            Q, R = np.linalg.qr(np.dot(M_arr[i], Q_prev))
            R_arr[i] = R
            Q_prev = Q
        M_arr = np.append(R_arr, [Q], axis = 0)

    print('Q needs to be close to identity, max difference is: ', 
           np.max(np.abs(np.diag(Q)) - np.abs(np.ones(Q.shape[0]))))
    diag = [np.diag(x) for x in R_arr]
    diag.append(np.diag(Q))
    diag = np.abs(diag)
    lRQR = np.sort(np.sum(np.log(diag), axis = 0)/(2*T))[::-1]
    return lRQR, Q

'''
Compute the local lyapunov expoenents (LLE) of a given system.

x0: point at which to calculate the LLE
f: dynamical system
fjac: jacobian
pf: parameters of system
pjac: parameters of jacobian
T: time over which to compute the LLE
L: number of steps
dt: integration time step

return local lyapunov exponents
'''
def LLE(x0, t0, f, fjac, pf, pjac, T, L, dt, num_it = 3):

    def dPhi_dt(Phi, t, x, pjac, Dim):
        """ The variational equation """
        rPhi = np.reshape(Phi, (Dim, Dim))
        rdPhi = np.dot(fjac(x, t, pjac), rPhi)
        return rdPhi.flatten()

    def dSdt(t, S, pf, pjac, Dim):
        """
        Differential equations for combined state/variational matrix
        propagation. This combined state is called S.
        
        p must have the dimension as the last element
        """
        x = S[:Dim]
        Phi = S[Dim:]
        return np.append(f(x, t, pf), dPhi_dt(Phi, t, x, pjac, Dim))

    Dim = len(x0)
    x_arr = np.float32(x0)
    Phi_arr = []
    tL = np.linspace(t0, t0+T, L+1, dtype=np.float32)
    
    Phi0 = np.eye(Dim, dtype=np.float32).flatten()
    for i,(t1,t2) in enumerate(zip(tL[:-1], tL[1:])):
        sol = solve_ivp(dSdt, [t1, t2],
                        np.append(x_arr, Phi0),
                        t_eval=[t1, t2],
                        method = 'RK45', #DOP853
                        max_step = dt,
                        args = (pf, pjac, Dim))
        x_arr = sol.y[:Dim].T[-1]
        Phi_arr.append(np.float32(sol.y[Dim:].T[-1].reshape(Dim, Dim)))
    Phi_arr = np.array(Phi_arr)
    OSE = np.concatenate((Phi_arr.transpose(0,2,1), np.flip(Phi_arr, axis = 0)))
    lyap, Q = rec_QR(OSE, T, num_it)
    gc.collect()
    return lyap