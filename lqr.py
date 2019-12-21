import numpy as np
from cartPlotter import cartPlotter
import control
from scipy.integrate import odeint
from stateValues import A,B,C,D,Q 
from responsePlot import responsePlot

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
	dydt = np.matmul((A-np.matmul(B,K_)),(y.reshape((-1,1))))
	return dydt.reshape(-1)

############## ODE solver for non linear model ##########
def pend_non_linear(y, t, A, B, K_):
	u = np.matmul(K_,y.reshape((-1,1)))[0,0]
	d1 = y[1]
	DD = M + m1 + m2 - m1*(np.cos(y[2])**2) - m2*(np.cos(y[4])**2) 
	d2 = -(1/DD)*(m1*g*np.sin(2*y[2])/2 // 
		+ m2*g*np.sin(2*y[4])/2 + m1*l1*y[3]*y[3]*np.sin(y[2]) // 
		+ m2*l2*y[5]*y[5]*np.sin(y[4]) + u) 
	d3 = y[3]
	d4 = (1/l1)*(d2*np.cos(y[2]) - g*np.sin(y[2]))
	d5 = y[5]
	d6 = (1/l2)*(d2*np.cos(y[4]) - g*np.sin(y[4]))
	return np.array([d1,d2,d3,d4,d5,d6])

############# Initial conditions ###########################
y0 = np.array([[0],[0],[-0.2],[0],[0.3],[0]])
t = np.arange(0, 100, 0.001)

############ Solves linear model with initial condition ###################
sol_linear = odeint(pend_linear, y0.reshape(-1), 
	t, args=(A, B, K[0]))
######### Solves non linear model with initial condition ###################
sol_non_linear = odeint(pend_non_linear, 
	y0.reshape(-1), t, args=(A, B, K[0]))

###################### shows the Animation ###############################

# print(np.linalg.eigvals(A - np.matmul()))

if run_animation:
	cartAnimatter = cartPlotter()
	cartAnimatter.run_animation(sol_linear, t, "Linear model", 2000)
	cartAnimatter.run_animation(sol_non_linear,t, "Non linear model", 2000)

response = responsePlot()
response.plotResponse(sol_linear, t, K, 
	"Response of linear system")
response.plotResponse(sol_linear, t, K, 
	"Response of non linear system")