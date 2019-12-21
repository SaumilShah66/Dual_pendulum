##############################  Start ################################
import numpy as np
from cartPlotter import cartPlotter
import control
from scipy.integrate import odeint
from stateValues import A,B,C,D,Q 
from responsePlot import responsePlot
import matplotlib.pyplot as plt

run_animation = 1 ### 0 if no animation is required
mean = 0  ### Mean od noise
std_dev = 0.16  ### Stdandard devication of noise 
m1, m2, M, l1, l2, g = 100, 100, 1000, 20, 10, 9.81

C = np.array([[1,0,0,0,0,0]])  ### measure x(t)

Q = np.array([[1/(12*1*1), 0, 10, 0, 0, 0],
	[0, 1/(12*0.1*0.1), 0, 0, 0, 0],
	[0, 0, 1/(8*0.02*0.02), 0, 0, 0],
	[0, 0, 0, 1/(12*0.01*0.01), 0, 0],
	[0, 0, 0, 0, 1/(8*0.02*0.02), 0],
	[0, 0, 0, 0, 0, 1/(12*0.01*0.01)]])
R = 0.00001

############## LQR gain ################################
K = control.lqr(A,B,Q,R)

############## ODE solver for non linear model ##########
def pend_non_linear(y, t, A, B, K_):
	u = np.matmul(K_,y.reshape((-1,1)))[0,0]
	d1 = y[1] + np.random.normal(mean, std_dev)
	DD = M + m1 + m2 - m1*(np.cos(y[2])**2) - m2*(np.cos(y[4])**2) 
	d2 = -(1/DD)*(m1*g*np.sin(2*y[2])/2 + m2*g*np.sin(2*y[4])/2  + m1*l1*y[3]*y[3]*np.sin(y[2]) + m2*l2*y[5]*y[5]*np.sin(y[4]) + u) + np.random.normal(mean, std_dev)
	d3 = y[3] + np.random.normal(mean, std_dev)
	d4 = (1/l1)*(d2*np.cos(y[2]) - g*np.sin(y[2])) + np.random.normal(mean, std_dev)
	d5 = y[5] + np.random.normal(mean, std_dev)
	d6 = (1/l2)*(d2*np.cos(y[4]) - g*np.sin(y[4])) + np.random.normal(mean, std_dev)
	return np.array([d1,d2,d3,d4,d5,d6])

############ ODE solver for Observer  ######## 
def obser(x_hat, t, L_, y_):
	u = -np.matmul(K[0],x_hat.reshape(-1,1))
	dxdt = np.matmul((A - np.matmul(L_,C)),x_hat.reshape(-1,1)) + B*u + np.matmul(L_,y_.reshape((-1,1)))  
	return dxdt.reshape(-1)

############# Solving non linear equation ############################
x0 = np.array([[0],[0],[-0.2],[0],[0.3],[0]])
t = np.arange(0, 100, 0.1)
####################################################################
L = control.place(np.matrix.transpose(A), 
	np.matrix.transpose(C), K[2])
L_ = np.matrix.transpose(L)
x_hat0 = x0.reshape(6)
u = -np.matmul(K[0],x_hat0)
observed = []
observed.append(x_hat0)

for i in range(len(t) - 1):
	print("solving "+str(i))
	X = odeint(pend_non_linear, x_hat0, t[i:i+2], args=(A, B, K[0]))
	y = np.matmul(C,(X[1]).reshape(-1,1)).reshape(-1) + np.random.normal(mean, std_dev)
	x_hat = odeint(obser , x_hat0 , t[i:i+2], args=(L_,y))
	x_hat0 = x_hat[1]
	u = -np.matmul(K[0],x_hat0)
	observed.append(x_hat0)
	
sol = np.array(observed)

if run_animation:
	cartAnimatter = cartPlotter()
	cartAnimatter.run_animation(sol, t[:sol.shape[0]], "Observer", 10)

response = responsePlot()
response.plotResponse(sol, t, K, 
	"Response of non linear system at noisy data")

############################  END  ############################################