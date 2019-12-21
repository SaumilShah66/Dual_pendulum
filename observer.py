import numpy as np
from cartPlotter import cartPlotter
import control
from scipy.integrate import odeint
from stateValues import A,B,C,D,Q 
from responsePlot import responsePlot
import matplotlib.pyplot as plt

########## For output state x(t) ##################
C = np.array([[1,0,0,0,0,0]])

####### For outout states x(t) and theta_2(t) ####
# C = np.array([[1,0,0,0,0,0],  
# 			[0,0,0,0,1,0]])

## For outout states x(t), theta_1(t) and theta_2(t) ####
# C = np.array([[1,0,0,0,0,0],
# 			[0,0,1,0,0,0],
# 			[0,0,0,0,1,0]])

run_animation = 1 ### 0 if no animation is required

m1, m2, M, l1, l2, g = 100, 100, 1000, 20, 10, 9.81

Q = np.array([[1/(12*1*1), 0, 10, 0, 0, 0],
	[0, 1/(12*0.1*0.1), 0, 0, 0, 0],
	[0, 0, 1/(8*0.02*0.02), 0, 0, 0],
	[0, 0, 0, 1/(12*0.01*0.01), 0, 0],
	[0, 0, 0, 0, 1/(8*0.02*0.02), 0],
	[0, 0, 0, 0, 0, 1/(12*0.01*0.01)]])
R = 0.00001

############## LQR gain ################################
K = control.lqr(A,B,Q,R)

############## ODE solver for linear model #############
def pend_linear(y, t, A, B, K_):
	dydt = np.matmul(A,(y.reshape((-1,1))))
	return dydt.reshape(-1)

############### ODE solver for non linear model ##########
def pend_non_linear(y, t, A, B, K_):
	d1 = y[1]
	DD = M + m1 + m2 - m1*(np.cos(y[2])**2) - m2*(np.cos(y[4])**2) 
	d2 = -(1/DD)*(m1*g*np.sin(2*y[2])/2 + m2*g*np.sin(2*y[4])/2 + m1*l1*y[3]*y[3]*np.sin(y[2]) + m2*l2*y[5]*y[5]*np.sin(y[4])) 
	d3 = y[3]
	d4 = (1/l1)*(d2*np.cos(y[2]) - g*np.sin(y[2]))
	d5 = y[5]
	d6 = (1/l2)*(d2*np.cos(y[4]) - g*np.sin(y[4]))
	return np.array([d1,d2,d3,d4,d5,d6])

############## ODE solver for observer #################
def obser(y, t, L_, y_):
	u = 0
	dxdt = np.matmul((A - np.matmul(L_,C)),y.reshape(-1,1)) + B*u + np.matmul(L_,y_.reshape((-1,1)))  
	return dxdt.reshape(-1)

##############  Initial conditions ###################
x0 = np.array([[0],[0],[-0.2],[0],[0.3],[0]])
t = np.arange(0, 100, 0.001)

############# Placing the poles of L at 3 times of poles of K #######
L = control.place(np.matrix.transpose(A), np.matrix.transpose(C), 
	3*K[2])
L_ = np.matrix.transpose(L)
x_hat0 = x0.reshape(6)
u = -np.matmul(K[0],x_hat0)
observed = []
error = []
trueStates = []
observed.append(x_hat0)

########### Loop to solve two differential equations ###########
########### One for plant and one for observer ###############
for i in range(len(t) - 1):
	if t[i] == 20:  ### Step input at t = 20 s
		x_hat0[0] = 1
	else:
		pass
	############### Plant linear or non linear ######
	X = odeint(pend_linear, x_hat0, t[i:i+2], args=(A, B, K[0]))
	trueStates.append(X[1])
	y = np.matmul(C,(X[1]).reshape(-1,1)).reshape(-1)
	############# Observer ###############
	x_hat_ = odeint(obser , x_hat0 , t[i:i+2], args=(L_,y))
	x_hat0 = x_hat_[1]  ## Initial condition for next step
	observed.append(x_hat0)
	error.append(X[1] - x_hat_[1])

trueStates = np.array(trueStates)
error = np.array(error)
states = np.array(observed)

if run_animation:
	cartAnimatter = cartPlotter()
	cartAnimatter.run_animation(states, t[:states.shape[0]], 
		"Linear model", 2000)

plt.plot(t[:error.shape[0]],error[:,0])
plt.title("Observer error for linear model")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.show()