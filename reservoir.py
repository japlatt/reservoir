import numpy as np
from scipy.stats import uniform
from scipy.integrate import odeint
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import lyapunov as lyap

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

        self.M = self.__getConnectionMat(self.N,
                                         a = params['M_a'],
                                         b = params['M_b'],
                                         pnz = params['M_pnz'],
                                         SR = params['M_SR'])

        self.Win = self.__getInputMat(self.N,
                                      self.D,
                                      a = params['Win_a'],
                                      b = params['Win_b'])
        self.system = params['system']

        self.Q = None
        self.Wout = None
        self.train_data = None

        self.train_t = None
        self.pred_t = None
        self.rmsd = None
        self.dQ = None
        self.Utrain = None

        self.LE = None
        self.LE_system = None
        self.LE_t = None

        self.pred_acc = None


    '''
    train the reservoir on data to find Wout

    trans_time = transient time to integrate to make sure reservoir synchronized
    train_time = time of the training window
    train_time_step = time step for finding Wout
    U = array of D, 1 dimensional interpolation functions valid for t in [-trans_time, train_time+pred_time]
    Q = function specifying the basis for u = Wout Q(r)
    beta = tikhenov regularization term
    constraints = double array of tuples giving the values of Wout to solve for
                  ex: [[(0, N)], [(0, N)], [(0, N//2), (N, 3*N//2)]]
    '''
    def train(self, train_time,
                    train_time_step,
                    Q,
                    beta,
                    constraints = None):

        assert(train_time_step >= self.time_step), 'Training time step must be > that integration time step'

        x0, self.Utrain = self.__spinup(train_time)
        train = odeint(self.__listening,
                       x0,
                       np.arange(0, train_time, self.time_step),
                       args = ((self.Utrain , self.gamma, self.M, self.Win),))


        self.train_t = np.arange(0, train_time, train_time_step)
        data = train[::int(train_time_step/self.time_step)]

        self.Q = Q
        self.Wout = self.__getWout(self.train_t, data, beta, constraints)
        self.train_data = np.dot(self.Wout, Q(data).T)
        return data

    '''
    Plot the training data to compare to the true system
    '''
    def plotTraining(self, show = True):
        assert(self.train_data is not None), 'must train the model first'
        u = np.array([x(self.train_t) for x in self.Utrain]) #construct u at time points needed

        fig, ax = plt.subplots(self.D, 1, sharex = True, figsize = (10, 7))
        for i in range(self.D):
            ax[i].plot(self.train_t, u[i], 'b-', linewidth = 2, label = 'True')
            ax[i].plot(self.train_t, self.train_data[i], 'r--',linewidth = 2, label = 'Estimate')
        ax[i].set_xlabel('Time', fontsize = 16)
        ax[0].set_title('Training Data: RMSD {:1.2f}'.format(self.rmsd), fontsize = 20)
        ax[i].legend(loc = 'upper right')
        if self.ifsave: fig.savefig(self.name+'_train.pdf', bbox_inches = 'tight')
        if show: plt.show()


    '''
    Use the predicting reservoir to predict forward in time.
    Plot the outcome.

    pred_time: time to check the prediction
    acc: accuracy for determining predictive power
    '''
    def predict(self, pred_time, acc = 1, show = True):
        assert(self.train_data is not None), 'must train the model first'

        x0, U = self.__spinup(pred_time)

        p = (self.gamma, self.M, self.Win, self.Wout, self.Q)
        self.pred_t = np.arange(0, pred_time, self.time_step)
        pred = odeint(self.__predicting,
                       x0,
                       self.pred_t,
                       args = (p,))

        u_pred = np.dot(self.Wout, self.Q(pred).T)
        u = np.array([x(self.pred_t) for x in U]) #construct u at time points needed

        if show:
            fig, ax = plt.subplots(self.D, 1, sharex = True, figsize = (10, 7))
            self.__plotPred(fig, ax, u, u_pred, self.pred_t, self.D)

        pred_acc = 0
        diff = np.sqrt(np.linalg.norm(u.T - u_pred.T, axis = 1)) > acc

        window = 10 #only triggers divergence if points in window > acc

        for i in range(len(diff)-window):
            if np.all([diff[i+j] for j in range(window)]):
                pred_acc = self.pred_t[i]
                break
        if show:
            for k in range(self.D): ax[k].axvline(pred_acc, linestyle = '--')
            if self.ifsave: fig.savefig(self.name+'_pred.pdf', bbox_inches = 'tight')
            plt.show()
        self.pred_acc = pred_acc-self.pred_t[0]
        return self.pred_acc





    '''
    Define the derivative of the Q function.  Needs to be only a function of r, the state
    of the reservoir.  For example

    if Q(r) = [r, r^2] then

    dQ = lambda r: np.concatenate((np.eye(len(r)), 2*np.diag(r)))
    '''
    def setDQ(self, dQ):
        self.dQ = dQ
            

    '''
    Compute the global lyapunov exponents using QR factorization method.

    time: as time-> large the accuracy of the lyap estimation increases
    dt: resolution over which to copute the lyap exp, make small to reveal
        fine structure in local variations
    system: system for which U was generated
    '''
    def globalLyap(self, time, dt):
        assert(self.train_data is not None), 'Need to train the reservoir to find Lyapunov Exponents'
        assert(self.dQ is not None), 'set DQ before running globalLyap'

        x0, U = self.__spinup(time)

        t = np.arange(0, time, dt)

        self.system.globalLyap(time, dt, x0 = np.array([u(0) for u in U]))

        p_list = (U, self.gamma, self.M, self.Win)
        p_jac = (self.Q, self.dQ, self.gamma, self.M, self.Win, self.Wout)
        LE = lyap.computeLE(self.__listening, self.__pred_jac, t,
                            min(self.time_step, dt/4), p_list, p_jac, x0, self.N)

        self.LE = LE
        self.LE_t = t

        return np.sort(self.system.LE[-1])[::-1], LE[-1]

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
            plt.plot(self.system.LE_t[1:], LE_system[:,i], 'r--')
        if self.ifsave: fig.savefig(self.name+'_LE.pdf', bbox_inches = 'tight')
        if show: plt.show()

    def getKYDim(self):
        assert(self.LE is not None), 'run globalLyap first'
        return lyap.KY_dim(self.LE)



    ###### PRIVATE FUNCTIONS ###############################


    '''
    synchronize the reservoir with the system.

    time: future time past the spinup time that we want U to be valid
    '''
    def __spinup(self, time):
        # integrate the system for spin up
        self.system.integrate(-self.trans_time, time)
        U = self.system.getU() 

        spinup = odeint(self.__listening,
                       np.random.rand(self.N),
                       np.arange(-self.trans_time, 0, self.time_step),
                       args = ((U, self.gamma, self.M, self.Win),))

        return spinup[-1], U
    '''
    plot the prediction of the reservoir
    '''
    @staticmethod
    def __plotPred(fig, ax, u, u_pred, t, D):
        for i in range(D):
            ax[i].plot(t, u[i], 'b-', linewidth = 2, label = 'True')
            ax[i].plot(t, u_pred[i], 'r--',linewidth = 2, label = 'Prediction')
        ax[i].set_xlabel('Time', fontsize = 16)
        ax[0].set_title('Predicting Data', fontsize = 20)
        plt.legend()

    '''
    Find Wout such that Wout Q(r) = u using linear regression.

    t = training time points
    data = training data
    beta = tikenov regularization
    constraints = double array of tuples giving the values of Wout to solve for
                  ex: [[(0, N)], [(0, N)], [(0, N//2), (N, 3*N//2)]]
    '''
    def __getWout(self, t, data, beta, constraints):
        dataQ = self.Q(data)

        #if constraints are not given then solve for all values
        if constraints == None:
            constraints = [[(0, self.N*dataQ.shape[1]//data.shape[1])] for x in range(self.D)]
        
        #initialize Wout
        Wout = np.zeros((self.D, self.N*dataQ.shape[1]//data.shape[1]), order = 'C')

        u = np.array([x(t) for x in self.Utrain]) #construct u at time points needed

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
        self.rmsd = self.__rmsd(dataQ, u, beta, Wout)
        return np.array(Wout, order = 'C')


    '''
    Find the root mean square deviation for the training
    '''
    @staticmethod
    def __rmsd(q, u, beta, Wout):
        z = np.dot(Wout, q.T)
        cst = np.linalg.norm(np.subtract(z,u)/len(u))
        return cst


    '''
    equations for the listening reservoir to be solved by solve_ivp
    dr/dt = gamma * (-r + tanh(M*r+Win*u))
    '''
    @staticmethod
    def __listening(r, t, p):
        U, gamma, M, Win = p
        MR = np.dot(M, r)
        WU = np.dot(Win, np.array([u(t) for u in U]))
        r = gamma * (-r + np.tanh(MR+WU))
        return r


    '''predicting reservoir
    dr/dt = gamma * (-r + tanh(M*r+Win*Wout*Q(r)))
    '''
    @staticmethod
    def __predicting(r, t, p):
        gamma, M, Win, Wout, Q = p
        u_pred = np.dot(Wout, Q(r).T)
        MR = np.dot(M, r)
        WU = np.dot(Win, u_pred)
        r = gamma * (-r + np.tanh(MR+WU))
        return r

    '''
    Jacobian of __predicting with respect to r
    '''
    @staticmethod
    def __pred_jac(r, t, p):
        Q, dQ, gamma, M, Win, Wout = p
        N = len(r)
        MR = np.dot(M, r)
        q = Q(r).T
        dq = dQ(r)
        WU = np.linalg.multi_dot((Win, Wout, q))
        S = np.diag(1 - np.tanh(MR+WU)**2)
        Wuhat = np.linalg.multi_dot((Win, Wout, dq))
        dfdr = gamma*(-np.eye(N)+np.dot(S, M+Wuhat))
        return dfdr


    '''
    Get the connection matrix for the reservoir

    N: Number of Neurons
    a: low bound on connections
    b: upper bound connections
    pnz: probability of element being nonzero
    SR: spectral radius or magnitude of principle eigenvalue, rescales a/b
    '''
    @staticmethod
    def __getConnectionMat(N, a = -1, b = 1, pnz = 0.02, SR = 0.9):
        #set elements in M from U(a, b) and enforce sparsity
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
    a: lower bound
    b: upper bound
    '''
    @staticmethod
    def __getInputMat(N, D, a = -1, b = 1):
        R = np.argmax(np.random.rand(N, D), axis = 1)
        mask = np.zeros((N, D))
        for i, j in enumerate(R):
            mask[i][j] = 1
        Win = np.multiply(np.random.uniform(a, b, (N, D)), mask)
        return np.array(Win, order = 'C')
