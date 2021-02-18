import numpy as np
from scipy.stats import uniform
from scipy.integrate import odeint
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import lyapunov as lyap
import seaborn as sns
from multiprocessing import Pool
from functools import partial

class reservoir:

    '''
    initialize the class

    params = {'name': name of the reservoir, used for saving files
              'N': size of the reservoir network,
              'D': dimension of the input system,
              'gamma': time const parameter,
              'M_a': lower bound on connection matrix M,
              'M_b': upper bound on connection matrix M,
              'M_pnz': sparsity of M (0-1),
              'M_SR': spectral radius of M (scales M_a/b),
              'Win_a': lower bound input matrix Win,
              'Win_b': upper bound input matrix Win,
              'time_step': time step for solving the model,
              'spinup_time': time it takes to synchronize res with sys
              'system': system to run reservoir over
              'ifsave': save images and files or not}
    '''

    def __init__(self, params):

        self.name = params['name']

        self.N = params['N']
        self.D = params['D']
        self.gamma = params['gamma']
        self.time_step = params['time_step']
        self.trans_time = params['spinup_time']

        self.ifsave = params['saveplots']

        self.M = self._getConnectionMat(self.N,
                                         a = params['M_a'],
                                         b = params['M_b'],
                                         pnz = params['M_pnz'],
                                         SR = params['M_SR'])

        self.Win = self._getInputMat(self.N,
                                     self.D,
                                     a = params['Win_a'],
                                     b = params['Win_b'],
                                     one_per_row = params.get('Win_one_per_row', True))

        self.system = params['system']
        self.train_noise = params.get('train_noise', 0)

        self.Q = None
        self.Wout = None
        self.train_data = None

        self.train_t = None
        self.pred_t = None
        self.cost = None
        self.dQ = None
        self.Utrain = None

        self.LE = None
        self.LE_system = None
        self.LE_t = None
        self.KY_dim = None

        self.pred_acc = None


    '''
    train the reservoir on data to find Wout

    train_time = time of the training window
    train_time_step = time step for finding Wout
    Q = function specifying the basis for u = Wout Q(r)
    beta = tikhenov regularization term
    constraints = double array of tuples giving the values of Wout to solve for
                  ex: [[(0, N)], [(0, N)], [(0, N//2), (N, 3*N//2)]]
    '''
    def train(self, train_time,
                    train_time_step,
                    Q,
                    beta,
                    constraints = None,
                    get_jac = False):

        assert(train_time_step >= self.time_step), 'Training time step must be > that integration time step'

        x0, self.Utrain, _ = self._spinup(train_time, noise = self.train_noise)
        train = odeint(self._listening,
                       x0,
                       np.arange(0, train_time, self.time_step, dtype = np.float32),
                       args = ((self.Utrain , self.gamma, self.M, self.Win),))


        self.train_t = np.arange(0, train_time, train_time_step, dtype = np.float32)
        data = train[::int(train_time_step/self.time_step)]

        self.Q = Q
        self.Wout = self._getWout(self.train_t, data, beta, constraints)
        self.train_data = np.dot(self.Wout, Q(data).T) if Q is not None else np.dot(self.Wout, data.T)

        if get_jac:
            Jac_est, Jac = self._get_train_jac(data)
            return data, Jac_est, Jac
        return data

    '''
    Plot the training data to compare to the true system
    '''
    def plotTraining(self, show = True):
        assert(self.train_data is not None), 'must train the model first'
        u = self.Utrain(self.train_t) #construct u at time points needed
        fig, ax = plt.subplots(self.D, 1, sharex = True, figsize = (10, 7))
        for i in range(self.D):
            ax[i].plot(self.train_t, u[i], 'b-', linewidth = 2, label = 'True')
            ax[i].plot(self.train_t, self.train_data[i], 'r--',linewidth = 2, label = 'Estimate')
        ax[i].set_xlabel('Time', fontsize = 16)
        ax[0].set_title('Training Data: cost {:1.2f}'.format(self.cost), fontsize = 20)
        ax[i].legend(loc = 'upper right')
        if self.ifsave: fig.savefig(self.name+'_train.pdf', bbox_inches = 'tight')
        if show: plt.show()


    '''
    get the difference between the jacobian of listening reservoir with respect to inputs
    and the jacobian of the system over the training time.
    '''
    # def getTrainJac(self, show = False):
    #     assert(self.Wout is not None), 'must train the model first'
    #     assert(self.system.fjac is not None), 'must provide system with Jacobian'

    #     if show:
    #         fig = plt.figure(figsize = (10, 7))
    #         plt.plot(self.train_t, self.train_jac_diff)
    #         if self.ifsave: fig.savefig(self.name+'_train_jac.pdf', bbox_inches = 'tight')
    #     return self.train_jac_diff


    '''
    Use the predicting reservoir to predict forward in time.
    Plot the outcome.

    pred_time: time to check the prediction
    acc: accuracy for determining predictive power
    '''
    def predict(self, pred_time, acc = None, show = False, retDat = False):
        assert(self.train_data is not None), 'must train the model first'
        if acc == None: acc = self.D

        x0, U, spinup = self._spinup(pred_time)

        self.predSysx0 = U(0)

        p = (self.gamma, self.M, self.Win, self.Wout, self.Q)
        self.pred_t = np.arange(0, pred_time, self.time_step, dtype = np.float32)
        pred = odeint(self.predicting,
                       np.float32(x0),
                       self.pred_t,
                       args = (p,))

        u_pred = np.dot(self.Wout, self.Q(pred).T) if self.Q is not None else np.dot(self.Wout, pred.T)

        u = U(self.pred_t)#construct u at time points needed
        pred_acc = self.pred_t[-1]
        diff = np.linalg.norm(u.T - u_pred.T, axis = 1)
        self.diff = diff
        cum_diff = np.cumsum(diff)/np.arange(1, len(diff)+1)
        diff = cum_diff > acc

        for i in range(len(diff)):
            if np.all(diff[i:]):
                pred_acc = self.pred_t[i]
                break

        self.pred_acc = pred_acc-self.pred_t[0]


        if show or self.ifsave:
            fig, ax = plt.subplots(self.D, 1, sharex = True, figsize = (10, 7))
            t_spin = np.arange(-min(self.trans_time/2, 10), 0, self.time_step)
            spin = np.dot(self.Wout, self.Q(spinup).T) if self.Q is not None else np.dot(self.Wout, spinup.T)
            spin = spin.T[int(len(spin.T) - len(t_spin)):]
            self._plotPred(fig, ax, U, u_pred, spin.T,
                            t_spin, self.pred_t, self.D)
            for k in range(self.D):
                ax[k].axvline(pred_acc, linestyle = '--')
                ax[k].tick_params(axis='both', labelsize=14)
                ax[k].set_ylabel(r'$x_{:d}$'.format(k), fontsize = 16)
            ax[-1].set_xlabel('Time', fontsize = 16)
            ax[0].set_title(self.name+' Prediction', fontsize = 20)
            if self.ifsave: fig.savefig(self.name+'_pred.png', bbox_inches = 'tight')
            if show: plt.show()
        if retDat: return self.pred_acc, u_pred, U(self.pred_t), pred
        return self.pred_acc


    '''
    Define the derivative of the Q function.  Needs to be only a function of r, the state
    of the reservoir.  For example

    if Q(r) = [r, r^2] then

    dQ = lambda r: np.concatenate((np.eye(len(r)), 2*np.diag(r)))
    '''
    def setDQ(self, dQ):
        self.dQ = dQ

    def getDQ(self, r):
        return self.dQ(r) if self.dQ is not None else None

    def getQ(self, r):
        return self.Q(r) if self.Q is not None else None
            
            

    '''
    ########### SLOW FUNCTION #############
    USE globalLyap_TLM instead

    Compute the global lyapunov exponents using QR factorization method.

    time: as time-> large the accuracy of the lyap estimation increases
    dt: resolution over which to copute the lyap exp, make small to reveal
        fine structure in local variations
    '''
    def globalLyap(self, time, dt):
        assert(self.train_data is not None), 'Need to train the reservoir to find Lyapunov Exponents'
        if self.Q is not None: assert(self.dQ is not None), 'set DQ before running globalLyap'

        x0, U, _ = self._spinup(time)

        t = np.arange(0, time, dt)

        self.system.globalLyap(time, dt, x0 = U(0))

        p_list = (U, self.gamma, self.M, self.Win)
        p_jac = (self.Q, self.dQ, self.gamma, self.M, self.Win, self.Wout)
        LE = lyap.computeLE(self._listening, self._pred_jac, t,
                            min(self.time_step, dt/4), p_list, p_jac, x0, self.N)

        self.LE = LE
        self.LE_t = t

        self.KY_dim = lyap.KY_dim(LE)

        return np.sort(self.system.LE[-1])[::-1], LE[-1]

    '''
    Compute the global lyapunov exponents using QR factorization method.  Uses
    a Tangent Linear Model (TLM) to solve the variational equations.  Should
    be much faster but maybe less accurate than globalLyap.

    time: as time-> large the accuracy of the lyap estimation increases
    dt: resolution over which to copute the lyap exp, make small to reveal
        fine structure in local variations
    num_save: number of exponents to save.  -1 means you save all of them.
    save_txt: save txt file of num_save exponents concurrently with calculation
    '''
    def globalLyap_TLM(self, time, dt, num_save = -1, savetxt = False):
        assert(self.train_data is not None), 'Need to train the reservoir to find Lyapunov Exponents'
        assert(self.dQ is not None), 'set DQ before running globalLyap'

        t = np.arange(0, time, self.time_step, dtype = np.float32)
        x, U = self.integrate(time)

        self.system.globalLyap(time, dt, x0 = U(0))

        pjac = (self.Q, self.dQ, np.float32(self.gamma), self.M,
                self.Win, self.Wout)

        rescale = int(dt/self.time_step)
        LE, self.KY_dim = lyap.computeLE_TLM(np.float32(x),
                                             t,
                                             self._pred_jac,
                                             pjac,
                                             self.name,
                                             rescale_interval = rescale,
                                             num_save = num_save,
                                             savetxt = savetxt)
        self.LE = LE
        self.LE_t = t[::rescale]

        return np.sort(self.system.LE[-1])[::-1], LE[-1]


    '''Conditional Lyapunov Exponents'''
    def CLE(self, time, dt):
        assert(self.train_data is not None), 'Need to train the reservoir to find Lyapunov Exponents'

        t = np.arange(0, time, self.time_step, dtype = np.float32)
        x, U = self.integrate(time)

        pjac = (U, self.gamma, self.Q, self.M, self.Win, self.Wout)
        
        rescale = int(dt/self.time_step)
        LE, KY_dim = lyap.computeLE_TLM( np.float32(x),
                                         t,
                                         self._list_jac,
                                         pjac,
                                         self.name,
                                         rescale_interval = rescale)
        return LE[-1]


    def localExp(self, x, t, U, L, dt, multi = False, num_it = 3):
        assert(len(x.shape) == 2), 'Provide multidimensional array'
        n, D = x.shape
        T = L*dt

        p = (U , self.gamma, self.M, self.Win)
        pjac = (self.Q, self.dQ, self.gamma, self.M, self.Win, self.Wout)
        

        if multi:
            with Pool() as pool: 
                parr_fun = partial(lyap.LLE, f = self._listening, fjac = self._pred_jac,
                                   pf = p, pjac = pjac, T = T, L = L, dt = self.time_step,
                                   num_it = num_it)
                LE= pool.starmap(parr_fun, zip(x, t))
                pool.close()
                pool.join()  

        else:
            LE = []
            for i, x0 in enumerate(x):
                LE.append(lyap.LLE(x0, t[i], self._listening, self._pred_jac,
                                  p, pjac, T, L, self.time_step, num_it))
        return np.array(LE)

    '''
    Plot the calculated Lyapunov exponents
    '''
    def plotLE(self, num_exp, show = True):
        assert(self.LE is not None), 'run globalLyap first'
        t = self.LE_t
        LE_system = self.system.LE
        fig,ax = plt.subplots(1,1,sharex=True) 
        lyap.plotNLyapExp(self.LE, t, num_exp, fig, ax)
        for i in range(max(2, np.sum(LE_system[-1] > -2))): 
            plt.plot(self.system.LE_t, LE_system[:,i], 'r--')
        if self.ifsave: fig.savefig(self.name+'_LE.pdf', bbox_inches = 'tight')
        if show: plt.show()

    '''
    Kaplan-Yorke dimension of the reservoir
    '''
    def getKYDim(self):
        assert(self.KY_dim is not None), 'run globalLyap first'
        return self.KY_dim


    def integrate(self, time):
        self.system.integrate(-2*self.trans_time, time+self.trans_time)
        U = self.system.getU()
        trans = odeint(self._listening,
                     np.random.rand(self.N),
                     np.arange(-self.trans_time, 0, self.time_step),
                     args = ((U, self.gamma, self.M, self.Win),))

        sol = odeint(self._listening,
                     trans[-1],
                     np.arange(0, time, self.time_step),
                     args = ((U, self.gamma, self.M, self.Win),))
        return sol, U

    def getListening(self):
        return self._listening

    def getPredicting(self):
        return self.predicting

    def getInpJac(self):
        return self._input_jac

    def getPredJac(self):
        return self._pred_jac

    def getSpinup(self):
        return self._spinup

    ###### PRIVATE FUNCTIONS ###############################


    '''
    synchronize the reservoir with the system.

    time: future time past the spinup time that we want U to be valid
    '''
    def _spinup(self, time, noise = 0):
        # integrate the system for spin up
        self.system.integrate(-2*self.trans_time, time, noise = noise)
        U = self.system.getU()

        spinup = odeint(self._listening,
                       np.random.rand(self.N).astype(np.float32),
                       np.arange(-self.trans_time, 0, self.time_step, dtype = np.float32),
                       args = ((U, self.gamma, self.M, self.Win),))

        return spinup[-1], U, spinup
    '''
    plot the prediction of the reservoir
    '''
    @staticmethod
    def _plotPred(fig, ax, U, u_pred, spinup, t_spin, t, D):
        palette = sns.color_palette('dark')
        u = U(t)
        u_spin = U(t_spin)
        for i in range(D):
            ax[i].plot(t_spin, u_spin[i], c = palette[0], lw = 2)
            ax[i].plot(t_spin, spinup[i], c = palette[1], ls = '--', linewidth = 2)
            ax[i].plot(t, u[i], c = palette[0], linewidth = 2, label = 'True')
            ax[i].plot(t, u_pred[i], c = palette[1], ls = '--', lw = 2, label = 'Prediction')
            ax[i].axvline(0, color = 'k', lw = 2)
        plt.legend()

    '''
    Find Wout such that Wout Q(r) = u using linear regression.

    t = training time points
    data = training data
    beta = tikenov regularization
    constraints = double array of tuples giving the values of Wout to solve for
                  ex: [[(0, N)], [(0, N)], [(0, N//2), (N, 3*N//2)]]
    '''
    def _getWout(self, t, data, beta, constraints):
        dataQ = self.Q(data) if self.Q is not None else data
        assert(dataQ.shape[0] == len(t))

        #if constraints are not given then solve for all values
        if constraints == None:
            constraints = [[(0, self.N*dataQ.shape[1]//data.shape[1])] for x in range(self.D)]

        self.constraints = constraints
        self._check_const(dataQ.shape[1]//self.N)
        
        #initialize Wout
        Wout = np.zeros((self.D, self.N*dataQ.shape[1]//data.shape[1]), order = 'C')

        u = self.Utrain(t) #construct u at time points needed

        #decompose matrix multiplication into D linear problems
        for i in range(self.D):
            #apply constraints to the data
            for j, c in enumerate(constraints[i]):
                if j == 0: data_mod = dataQ.T[c[0]:c[1]]
                else: data_mod = np.concatenate((data_mod, dataQ.T[c[0]:c[1]]))

            #solve the linear problem
            rrt_inv = np.linalg.inv(np.dot(data_mod, data_mod.T)+beta*np.eye(data_mod.shape[0]))
            ui = u[i]
            Wi = np.linalg.multi_dot((rrt_inv, data_mod, ui))

            #revert back to regular matrix size
            for k, c in enumerate(constraints[i]):
                if k == 0: start = 0
                else: start = end 
                end = start+c[1]-c[0]
                Wout[i, c[0]:c[1]] = Wi[start:end]
        self.cost = self._cost(dataQ, u, beta, Wout)
        return np.array(Wout, order = 'C', dtype = np.float32)


    '''
    Find the root mean square deviation for the training
    '''
    @staticmethod
    def _cost(q, u, beta, Wout):
        z = np.dot(Wout, q.T)
        cst = np.linalg.norm(np.subtract(z,u))**2/u.size
        return cst


    '''
    equations for the listening reservoir to be solved by solve_ivp
    dr/dt = gamma * (-r + tanh(M.r+Win.u))
    '''
    @staticmethod
    def _listening(r, t, p):
        U, gamma, M, Win = p
        MR = M.dot(r)
        WU = Win.dot(U(t))
        r = gamma * (-r + np.tanh(MR+WU))
        return r

    '''Jacobian of _listening with respect to u in the dimension of u'''
    @staticmethod
    def _input_jac(r, t, p):
        U, gamma, Q, M, Win, Wout = p
        MR = M.dot(r)
        WU = Win.dot(U(t))
        S = 1 - np.tanh(MR+WU)**2
        J = []
        for s in S.T:
            sW = sparse.diags(s).dot(Win)
            J.append(np.dot(Wout, sW))
        return np.array(J)

    '''Jacobian of _listening with respect to r'''
    @staticmethod
    def _list_jac(r, t, p):
        U, gamma, Q, M, Win, Wout = p
        N = len(r)
        MR = M.dot(r)
        WU = Win.dot(U(t))
        S = sparse.diags(1 - np.tanh(MR+WU)**2)
        dfdr = gamma*(-np.eye(N)+S.dot(M))
        return dfdr.astype(np.float32)


    '''
    get the mapped jacobian with respect to u for the listening reservoir and compare to the
    actual jacobian of the dynamical system in the dimension of u.
    '''
    def _get_train_jac(self, data):
        assert(self.system.fjac is not None), 'set fjac for system'
        Jac_est = self._input_jac(data.T, self.train_t,
                                   (self.Utrain, self.gamma,
                                    self.Q, self.M, self.Win, self.Wout))

        X = self.Utrain(self.train_t)
        Jac = np.zeros((len(self.train_t), self.D, self.D))
        for i in range(len(self.train_t)):
            Jac[i] = self.system.fjac(X[:, i], self.train_t[i], self.system.p)
        return Jac_est, Jac

    '''
    predicting reservoir
    dr/dt = gamma * (-r + tanh(M.r+Win.Wout.Q(r)))
    '''
    @staticmethod
    def predicting(r, t, p):
        gamma, M, Win, Wout, Q = p
        u_pred = np.dot(Wout, Q(r).T) if Q is not None else Wout.dot(r)
        MR = M.dot(r)
        WU = Win.dot(u_pred)
        r = gamma * (-r + np.tanh(MR+WU))
        return r

    '''
    Jacobian of predicting with respect to r
    '''
    @staticmethod
    def _pred_jac(r, t, p):
        Q, dQ, gamma, M, Win, Wout = p
        N = len(r)
        MR = M.dot(r)
        q = Q(r) if Q is not None else r
        dq = dQ(r) if dQ is not None else np.eye(len(r))
        # WU = np.linalg.multi_dot((Win, Wout, q))
        WU = Win.dot(np.dot(Wout, q))
        S = sparse.diags(1 - np.tanh(MR+WU)**2)
        # Wuhat = np.linalg.multi_dot((Win, Wout, dq))
        Wuhat = Win.dot(np.dot(Wout, dq))
        dfdr = gamma*(-np.eye(N, dtype = np.float32)+S.dot(M+Wuhat))
        return dfdr.astype(np.float32)


    '''
    Get the connection matrix for the reservoir

    N: Number of Neurons
    a: low bound on connections
    b: upper bound connections
    pnz: probability of element being nonzero
    SR: spectral radius or magnitude of principle eigenvalue, rescales a/b
    '''
    @staticmethod
    def _getConnectionMat(N, a = -1, b = 1, pnz = 0.02, SR = 0.9):
        #set elements in M from U(a, b) and enforce sparsity
        M = sparse.random(N, N, density = pnz, data_rvs = uniform(loc=a, scale=(b-a)).rvs)
        M = sparse.csr_matrix(M, dtype = np.float32)
        
        #rescale to specified spectral radius
        SRM = np.abs(sparse.linalg.eigs(M, k = 1, which='LM', return_eigenvectors=False))
        M = M.multiply(SR/SRM)
        return M #np.squeeze(np.asarray(M.todense(), order = 'C'))



    '''
    Get the input matrix for the reservoir

    N: number of neurons
    D: dimensionality of the input
    a: lower bound
    b: upper bound
    '''
    @staticmethod
    def _getInputMat(N, D, a = -1, b = 1, one_per_row = True):
        if one_per_row:
            R = np.argmax(np.random.rand(N, D), axis = 1)
            mask = np.zeros((N, D))
            for i, j in enumerate(R):
                mask[i][j] = 1
        else:
            return np.random.uniform(a, b, (N, D))
        Win = np.multiply(np.random.uniform(a, b, (N, D)), mask)
        return sparse.csr_matrix(Win, dtype = np.float32)



    def _check_const(self, q_shape):
        ''' check the constraints to make sure they are consistent '''
        for i in range(self.D):
          c_prev = 0
          for j, c in enumerate(self.constraints[i]):
            c0 = c[0]
            c1 = c[1]
            assert(c1 > c0)
            assert(c0 >= c_prev)
            assert(c1 <= self.N*q_shape)

            c_prev = c1
