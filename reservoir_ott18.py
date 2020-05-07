import numpy as np
from scipy.integrate import odeint
import scipy as sp
import matplotlib.pyplot as plt
from Res_Ott18_helper import *
import scipy.optimize as optimize
import time

dt = 0.001 #0.001 in paper
start = -100 #-100
tot_time = 160 #160

U, u, D, t = getInputData('lorenz63',np.random.rand(3), dt, start, tot_time)

#size of reservoir
N = 2000 #N = 2000 in paper

M = getConnectionMat(N, pnz = 0.02, SR = 0.9)
Win = getInputMat(N, D)

np.save('data/M.npy', M)
np.save('data/Win.npy', Win)



#constants reservoir
scale = 0.014 #sigma
gamma = 10

#propagate the listening reservoir

r = odeint(listening, np.random.rand(N), t, args = (U, gamma, scale, M, Win))

#cut off transients so starts from t = 0
t0 = int(abs(start)/dt)
rp = r[t0: , :]
up = u[t0: , :]

'''q is the 2N-dimensional vector consisting of the N coor of r 
follows by their squares'''
qp = Q(rp)


skip = 20 #sparsely sample to speed up optimization
qp_skip = np.array(qp[::skip, :].T, order = 'C')
up_skip = up[::skip, :].T
rp_skip = np.array(rp[::skip, :].T, order = 'C')


print('Solving for Wout')
startTime = time.time()
beta = 1e-6
Wout = np.zeros((D, 2*N), dtype = np.float32, order = 'C')

qmod = np.concatenate((qp_skip[:N//2], qp_skip[3*N//2:]))

rrt_inv = np.linalg.inv(np.dot(rp_skip, rp_skip.T)+beta*np.eye(N))
qqt_inv = np.linalg.inv(np.dot(qmod, qmod.T)+beta*np.eye(N))
ux, uy, uz = up_skip

Wx = np.linalg.multi_dot((rrt_inv, rp_skip, ux))
Wy = np.linalg.multi_dot((rrt_inv, rp_skip, uy))
Wz = np.linalg.multi_dot((qqt_inv, qmod, uz))

Wout[0, :N] = Wx
Wout[1, :N] = Wy
Wout[2, :N//2] = Wz[:N//2]
Wout[2, 3*N//2:] = Wz[N//2:]
print('Optimization Time: ',time.time()-startTime)
print('Cost: ', cost(qp_skip, up_skip, beta, Wout))

np.save('data/Wout.npy', Wout)



#PREDICTION
#---------------------------------------------------------------------------------
time_pred = 35
_, u_true, _, tpred = getInputData('lorenz63', up[-1], dt, start+tot_time, time_pred)

startt = t0+int(50/dt)
startu = int(50/dt)

r_true = odeint(predicting, rp[-1], tpred, args = (gamma, scale, M, Win, Wout))

u_pred = np.dot(Wout, Q(r_true).T)
u_est = np.dot(Wout, Q(rp[startu:]).T)


#Plot results
fig, ax = plt.subplots(3, 1, sharex = True, figsize = (10, 7))

fig.suptitle('Reservoir Lorenz 63 Esimation and Prediction', fontsize = 20)

ax[0].plot(t[startt:], u_est[0], 'b--', linewidth = 2)
ax[0].plot(t[startt:], up.T[0][startu:], 'r-', alpha = 0.5, linewidth = 2)
ax[0].plot(tpred, u_pred[0], 'b--', linewidth = 2, label = 'prediction')
ax[0].plot(tpred, u_true.T[0], 'r-', alpha = 0.5, linewidth = 2, label = 'true')
ax[0].axvline(t[-1], color = 'k', linewidth = 4)
ax[0].set_ylabel('Lorenz 63 X', fontsize = 16)

ax[1].plot(t[startt:], u_est[1], 'b--', linewidth = 2)
ax[1].plot(t[startt:], up.T[1][startu:], 'r-', alpha = 0.5, linewidth = 2)
ax[1].plot(tpred, u_pred[1], 'b--', linewidth = 2, label = 'prediction')
ax[1].plot(tpred, u_true.T[1], 'r-', alpha = 0.5, linewidth = 2, label = 'true')
ax[1].axvline(t[-1], color = 'k', linewidth = 4)
ax[1].set_ylabel('Lorenz 63 Y', fontsize = 16)

ax[2].plot(t[startt:], u_est[2], 'b--', linewidth = 2)
ax[2].plot(t[startt:], up.T[2][startu:], 'r-', alpha = 0.5, linewidth = 2)
ax[2].plot(tpred, u_pred[2], 'b--', linewidth = 2, label = 'prediction')
ax[2].plot(tpred, u_true.T[2], 'r-', alpha = 0.5, linewidth = 2, label = 'true')
ax[2].axvline(t[-1], color = 'k', linewidth = 4, label = 'prediction boundary')
ax[2].set_ylabel('Lorenz 63 Z', fontsize = 16)

plt.legend()
plt.show()

plt.savefig('data/Ott18_fig5.pdf', bbox_inches = 'tight')




