import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

class rulkov_map:
	def __init__(self, alpha, mu, sigma, xstart, ystart):
		self.alpha = alpha
		self.mu = mu
		self.sigma = sigma
		self.x = xstart
		self.y = ystart
		self.xlist = [xstart]

	def calc_f(self):
		if self.x <= 0:
			f = self.alpha/(1-self.x)+ self.y
		elif self.x < self.alpha + self.y:
			f = self.alpha + self.y
		else:
			f = -1
		return f

	def update(self):
		xnew = self.calc_f()
		ynew = self.y - self.mu*(self.x + 1) + self.mu*self.sigma
		self.x = xnew
		self.y = ynew
		self.xlist.append(self.x)

	def simulate(self,num_iter):
		for i in tqdm(range(num_iter)):
			self.update()


def plot_x(x):
	plt.plot(range(0,len(x)), x)
	plt.axis([0, len(x)+1, min(x)-1, max(x)+1])
	plt.ylabel("X_n (Membrane Potential")
	plt.xlabel("Iteration number N")
	plt.show()


def main():
	choice = sys.argv[1]

	# silence
	if choice == 'S':
		rm = rulkov_map(alpha=4, mu=0.001, sigma=-0.01, xstart=0.5, ystart=-2.8)

	# tonic spiking
	# the frequesncy of spiking increases with increasing sigma
	elif choice == 'T':
		rm = rulkov_map(alpha=4, mu=0.001, sigma=0.01, xstart=1, ystart=-3)

	# burst spiking
	# alpha > 4
	elif choice == 'B':
		rm = rulkov_map(alpha=6, mu=0.001, sigma=0.14, xstart=-1, ystart=-4)
	
	else:
		print "Wrong Choice Entered!\nPlease enter S:Silence, T:Tonic or B:Burst.\nAborting!"
		exit(0)

	# rm = rulkov_map(alpha=4.5, mu=0.001, sigma=0.5, xstart=-1, ystart=-4)

	

	rm.simulate(2000)
	plot_x(rm.xlist)

if __name__ == "__main__":
	main()
	# https://arxiv.org/pdf/nlin/0201006.pdf