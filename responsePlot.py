import matplotlib.pyplot as plt
import numpy as np


class responsePlot:
	def __init__(self):
		pass

	def plotResponse(self, sol, t, K, title):
		fig2 = plt.figure()
		fig2.suptitle(title, fontsize=16)
		ax_1 = fig2.add_subplot(311)
		ax_2 = fig2.add_subplot(312)
		ax_u = fig2.add_subplot(313)
		ax_u.set_xlabel("Time in Seconds")
		ax_u.set_ylabel("Force in Newtons")
		ax_u.set_title("Input force to stabilize system")
		ax_1.set_title("Response of X, theta_1 and theta_2")
		ax_2.set_title("Response of X_dot, theta_1_dot and theta_2_dot")
		ax_1.plot(t,sol[:,0].reshape(-1), label="X in meters")
		ax_1.plot(t,sol[:,2].reshape(-1), label="theta_1 in radians")
		ax_1.plot(t,sol[:,4].reshape(-1), label="theta_2 in radians")
		ax_2.plot(t,sol[:,1].reshape(-1), label="X_dot")
		ax_2.plot(t,sol[:,3].reshape(-1), label="theta_1_dot")
		ax_2.plot(t,sol[:,5].reshape(-1), label="theta_2_dot")
		ax_1.legend()
		ax_2.legend()
		U_ = np.matmul(K[0],np.matrix.transpose(sol))
		ax_u.plot(t, U_.reshape(-1))
		plt.show()