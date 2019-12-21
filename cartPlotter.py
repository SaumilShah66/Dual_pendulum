import matplotlib.pyplot as plt
import numpy as np

class cartPlotter:
	def __init__(self):
		pass

	def plot_cart(self,x_, mass, ax):
		den = 100
		l = 1
		ax.plot([x_ + l/2, x_ + l/2, x_ - l/2, x_ - l/2, x_ + l/2], 
			[0, l, l, 0, 0] )
		pass

	def ball(self,theta, l, x_, mass, ax):
		l = l/10
		ax.scatter(x_ - l*np.sin(theta), -l*np.cos(theta), s=mass*2)
		ax.plot([x_, x_ - l*np.sin(theta)], [0, -l*np.cos(theta)])
		pass

	def write_vals(self,sol, i, t, ax):
		ax.text(0.8,0.95,'x : '+str(round(sol[i,0],2)),
			horizontalalignment='left',verticalalignment='center',
			transform=ax.transAxes)
		ax.text(0.8,0.9,'dx/dt : '+str(round(sol[i,1],2)),
		 	horizontalalignment='left',verticalalignment='center', 
		 	transform=ax.transAxes)
		ax.text(0.8,0.85,'t1 : '+str(round(sol[i,2],2)), 
			horizontalalignment='left',verticalalignment='center', 
			transform=ax.transAxes)
		ax.text(0.8,0.8,'d(t1)/dt : '+str(round(sol[i,3],2)), 
			horizontalalignment='left',verticalalignment='center', 
			transform=ax.transAxes)
		ax.text(0.8,0.75,'t2 : '+str(round(sol[i,4],2)), 
			horizontalalignment='left',verticalalignment='center', 
			transform=ax.transAxes)
		ax.text(0.8,0.7,'d(t2)/dt : '+str(round(sol[i,5],2)), 
			horizontalalignment='left',verticalalignment='center', 
			transform=ax.transAxes)
		ax.text(0.8,0.65,'time : '+str(round(t[i],2)), 
			horizontalalignment='left',verticalalignment='center', 
			transform=ax.transAxes)
	
	def run_animation(self, sol, t, title, rate):
		m1, m2, M, l1, l2, g = 100, 100, 1000, 20, 10, 9.81
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.ion()
		for i in range(0,sol.shape[0],rate):
			if i%10==0:
				print(" Number : "+str(i))
				pass
			self.plot_cart(sol[i,0], M, ax)
			self.ball(sol[i,2], l1, sol[i,0], m1, ax)
			self.ball(sol[i,4], l2, sol[i,0], m2, ax)
			self.write_vals(sol, i, t, ax)
			ax.plot([-50,50],[0,0],c="black")
			plt.xlim(-5, 5)
			plt.ylim(-10, 10)
			plt.title(title)
			plt.pause(0.0001)
			plt.cla()

		plt.ioff()  
		self.plot_cart(sol[i,0], M, ax)
		self.ball(sol[i,2], l1, sol[i,0], m1, ax)
		self.ball(sol[i,4], l2, sol[i,0], m2, ax)
		self.write_vals(sol, i, t, ax)
		ax.plot([-5,5],[0,0],c="black")
		plt.xlim(-5, 5)
		plt.ylim(-10, 10)
		plt.title(title)
		plt.show()