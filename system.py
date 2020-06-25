import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import scipy as sp
import lyapunov as lyap
from multiprocessing import Pool
from functools import partial


class system:

    '''
    Initialize the system

    f: dynamical system (needs to be in odeint format f(x, t, args*))
    p: parameters for f
    D: dimension of the system
    fjac: jacobian of f, needed for computation of lyapunov exponents
    '''
    def __init__(self, f, p, D, dt, fjac = None):
        self.f = f
        self.fjac = fjac
        self.p = p
        self.D = D
        self.dt = dt
        self.U = None
        self.T = None
        self.LE = None
        self.min_emb = None
        self.t = np.arange(0, 1000*dt, dt)
        self.u = odeint(self.f, np.random.rand(self.D), self.t,
                   args = (self.p,))


    '''
    Integrate system forward in time using odeint.  Needed to run anything else.

    tstart: start time
    tend: end time
    tstep: time step for integration
    '''
    def integrate(self, tstart, tend, tstep = None):
        assert(tstart < tend), 'start needs to be before end'
        if tstep is not None: self.dt = tstep
        x0 = self.u[np.random.choice(self.u.shape[0], 1, replace=False)].squeeze()
        x0 += np.random.rand(self.D)
        t = np.arange(tstart, tend, self.dt)
        u = odeint(self.f, x0, t,
                   args = (self.p,))
        self.t = t
        self.u = u

        return t, u



    '''
    Function to get interpolated function of the solution to the dynamical system.
    This is the function that will be passed to the reservoir.

    Must have run integrate first
    '''
    def getU(self):
        assert(self.u is not None), 'U has not been set, run integrate function first'
        u_arr = []
        for i in range(self.D):
            u_arr.append(interp1d(self.t, self.u[:, i], fill_value="extrapolate"))
        self.U = u_arr
        return self.U


    '''
    Function to find the first minimum of the average mutual information (AMI) of a time delayed
    system.  This will give us the optimal time delay for time delay embedding.

    time_lag: compute the mutual information of time delays from 1...time_lag/dt.
    u_ind: since we only need a single time series for this analysis, pick 1
    num_bins: number of bins to compute mutual information.  If none then use
              sturges rule to pick.

    return: time delay time step that gives minimum of mutual information

    user input required to specify the range to search for the first minimum of
    the mutual information.

    Ref: Analysis of Observed Chaotic Data, Abarbanel 1996
    '''
    def findMinAMI(self, time_lag, u_ind = 0, num_bins = None):
        assert(self.u is not None), 'run integrate function first'
        lag = np.arange(1, int(time_lag/self.dt)+1)
        ui = self.u[:, u_ind]
        MI = np.zeros(len(lag))
        if num_bins == None: num_bins = int(np.round(np.log2(len(ui))+1)) #Sturges rule
        for l in lag: MI[l-1] = self.__calc_MI(ui[:-l], ui[l:], num_bins)
        


        fig = plt.figure(figsize = (8,5))
        plt.plot(lag, MI, 'b-')
        plt.xlabel('Timelag')
        plt.ylabel('MI')
        fig.suptitle('Average Mutual Information')
        plt.show()


        print('Enter a number up to search for first minimum')
        end_search = int(input('end search: '))
        self.T = np.argmin(MI[0:end_search])
        print('The first min of MI is: T={:d}'.format(self.T))
        return self.T


    '''
    State Space Reconstruction

    False nearest neighbors gives the minimum embedding dimension for the
    time series given.  This is not necessarily the dimension of the original
    system.
    
    nDim: number of dimensions to search over.  We are looking for the number
          of dimensions for which the attractor is fully unfolded which is 
          when the number of false nearest neighbors is 0.

    Ref: Analysis of Observed Chaotic Data, Abarbanel 1996
    '''
    def FNN(self, nDim, u_ind = 0):
        assert(self.u is not None), 'run integrate function first'
        assert(self.T is not None), 'call findMinAMI before running FNN'
        NFNN = np.zeros(nDim)
        x = self.u[:, u_ind]
        y = x[:-nDim*self.T-1]
        for i in range(nDim):
            y2 = np.vstack((y, x[(i+1)*self.T:-self.T*nDim+(i+1)*self.T-1]))

            if i == 0:
                X = y.reshape(1, -1)
                Tree = sp.spatial.cKDTree(X.T)
                dist, inds = Tree.query(X.T, k = 2)

                dist = dist[:,1]**2
                inds = inds[:, 1]

            else:
                Tree = sp.spatial.cKDTree(y.T)
                dist, inds = Tree.query(y.T, k = 2)

                dist = dist[:,1]**2
                inds = inds[:, 1]

            dist2 = np.array([np.linalg.norm(y2.T[i] - y2.T[inds[i]], 2)**2 for i in range(len(y.T))])

            f = np.divide(np.abs(dist2-dist),dist)
            f = f > 15**2
            NFNN[i] = np.sum(f)/(1.0*len(f))
            y = y2

        print(np.round(NFNN,3)*100)
        for i, n in enumerate(NFNN):
            if n < 1e-3: 
                print('minimum embedding dimension is {:d} with NFNN {:1.3f}'.format(i+1, n*100))
                self.min_emb = i+1
                return
        print('ERROR: need to search more dimensions')
        


    '''
    Compute the global lyapunov exponents using QR factorization method.

    time: as time-> large the accuracy of the lyap estimation increases
    dt: resolution over which to copute the lyap exp, make small to reveal
        fine structure in local variations
    x0: point at which to start computation of global exp
    '''
    def globalLyap(self, time, dt, x0 = None):
        assert(self.fjac is not None), 'Need to provide jacobian for lyapunov expononents'
        if x0 is None:
            assert(self.u is not None), 'run integrate function first'
            x0 = self.u[-1]
        t = np.arange(0, time, dt)
        
        LE = lyap.computeLE(self.f, self.fjac, t, self.dt, self.p, self.p, x0, self.D)

        self.LE = LE
        self.LE_t = t

        return LE[-1]

    '''
    plot num_exp number of exponents over
    '''
    def plotLE(self, num_exp):
        assert(self.LE is not None), 'run globalLyap first'
        LE = self.LE
        t = self.LE_t
        fig,ax = plt.subplots(1,1,sharex=True) 
        lyap.plotNLyapExp(LE, t, num_exp, fig, ax)
        plt.show()


    '''
    Kaplan-Yorke dimension for system
    '''
    def getKYDim(self):
        assert(self.LE is not None), 'run globalLyap first'
        return lyap.KY_dim(self.LE)


    '''
    Calculate the finite time lyapunov exponents at points x

    x: 2D array of points at which to calculate FTLE
    L: number of steps ahead to calculate
    dt: time step so T = L*dt where T is the time going forward
    multi: multiprocessing true or false
    '''
    def localExp(self, x, L, dt, multi = False):
        assert(self.fjac is not None), 'Need to provide jacobian for lyapunov expononents'
        assert(len(x.shape) == 2), 'Provide multidimensional array'
        n, D = x.shape
        assert(D == self.D)
        T = L*dt

        if multi:
            with Pool() as pool: 
                parr_fun = partial(lyap.LLE, f = self.f, fjac = self.fjac,
                                   pf = self.p, pjac = self.p, T = T, L = L, dt = self.dt)
                LE= pool.map(parr_fun, x)
                pool.close()
                pool.join()  

        else:
            LE = []
            for i, x0 in enumerate(x):
                LE.append(lyap.LLE(x0, self.f, self.fjac,
                                 self.p, self.p, T, L, self.dt))
        return np.array(LE)


    ########## PRIVATE FUNCTIONS ####################
    '''
    Calculate mutual information of 2 time series X, Y
    '''
    @staticmethod
    def __calc_MI(X, Y, bins):
        c_xy = np.histogram2d(X, Y, bins)[0]
        MI = mutual_info_score(None, None, contingency=c_xy)
        return MI