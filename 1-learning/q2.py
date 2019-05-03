import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

'''
Command Line inputs to the code:
--------------------------------------
1. n = Grid size (nxn)
2. m = number of holes in the grid
--------------------------------------
'''

def feq(a,b):
    if abs(a-b)<0.000001:
        return 1
    else:
        return 0

def softmax(a, beta=1.0):
    _a = a-np.max(a)
    b = np.exp(beta*_a)
    return b/np.sum(b)

class frozen_lake:
	def __init__(self, n, m):
		# q : q matrix (n x n x a)
		# r : reward
		# a : 0=U, 1=R, 2=D, 3=L
		self.n = n
		self.m = m

		self.q = [ [ [ 0 for i in range(4) ] for j in range(self.n) ] for k in range(self.n) ]

		# a[i,j] is the set of possible actions available at i,j 
		self.a = [ [ [0,1,2,3] for i in range(self.n) ] for j in range(self.n) ]

		for i in range(self.n):
			self.a[0][i].remove(0)
			self.a[i][n-1].remove(1)
			self.a[n-1][i].remove(2)
			self.a[i][0].remove(3)

		# creating an instance of the frozen lake
		# assigned a reward of +1 to the goal and -1 to each of the holes
		self.r = [ [ 0 for i in range(self.n) ] for j in range(self.n) ]

		# creating m holes randomly and assigning them reward of -1
		num_holes_created = 0
		while num_holes_created < self.m:
			i = random.choice(range(self.n))
			j = random.choice(range(self.n))
			if self.r[i][j] == 0 and not (i==0 and j==0) and not(i==n-1 and j==n-1):
				self.r[i][j] = -1
				num_holes_created += 1

		# the start point is (0,0) and the end point is (n-1,n-1)
		self.s = [0,0]
		self.g = [self.n-1,self.n-1]

		# assigning reward to goal
		self.r[self.g[0]][self.g[1]] = 1

		# no further actions to be taken at the goal
		self.a[self.g[0]][self.g[1]] = []


	def init_learning_params(self):
		self.max_epis = 10000
		self.alpha = 0.6
		self.lambdaa = 0.9
		self.epsilon = 0.5 
		self.epsilonDecay = 0.99
		self.max_steps = 200
		self.beta = 0.0
		self.beta_inc = 0.02
		self.action_policy = 'epsilon_greedy'
		# self.action_policy = 'softmax'

	# e-greedy approach for selecting action
	def select_action_epsilon_greedy(self, state):
		i = state[0]
		j = state[1]
		if random.random() < self.epsilon:
			action = random.choice(self.a[i][j])
		else:
			best_actions = []
			max_q = min(self.q[i][j])
			# find the max possible q value
			for k in self.a[i][j]:
				if self.q[i][j][k] > max_q:
					max_q = self.q[i][j][k]
			# get all possible actions that achieve that q value
			for k in self.a[i][j]:
				if feq(self.q[i][j][k], max_q):
					best_actions.append(k)
			# choose randomly from among those actions
			action = random.choice(best_actions)
		self.epsilon = self.epsilon*self.epsilonDecay
		return action

	# softmax approach to select action
	def select_action_softmax(self, state):
		i = state[0]
		j = state[1]
		probs = softmax(self.q[i][j], beta=self.beta)
		while True:
			action = np.random.choice(4, p=probs)
			if action in self.a[i][j]:
				break
		return action


	# get the next state from the given state and action
	def get_next(self, state, action):
		i = state[0]
		j = state[1]
		next_state = []
		if action == 0: 
			next_state = [i-1, j]
		elif action == 1:
			next_state = [i, j+1]
		elif action == 2:
			next_state = [i+1, j]
		elif action == 3:
			next_state = [i, j-1]
		reward = self.r[next_state[0]][next_state[1]]
		return next_state, reward

	# update q matrix for the given state, action and next state
	def update_q(self, curr_state, next_state, action, reward):
		i = curr_state[0]
		j = curr_state[1]
		curr_q = self.q[i][j][action]
		future_pred = min(self.q[next_state[0]][next_state[1]])
		for future_act in self.a[next_state[0]][next_state[1]]:
			if self.q[next_state[0]][next_state[1]][future_act] > future_pred:
				future_pred = self.q[next_state[0]][next_state[1]][future_act]

		self.q[i][j][action] = curr_q + self.alpha*(reward + self.lambdaa*future_pred - curr_q)


	# q learning
	def q_learn(self):
		self.init_learning_params()
		rewards = []
		for epis in tqdm(range(self.max_epis)):
			# curr_state = self.s
			curr_state = [random.choice(range(self.n)), random.choice(range(self.n))]
			total_reward = 0

			for curr_step in range(self.max_steps):
				if curr_state == self.g:
					break

				if self.action_policy == 'epsilon_greedy':
					action = self.select_action_epsilon_greedy(curr_state)
				elif self.action_policy == 'softmax':	
					action = self.select_action_softmax(curr_state)
				next_state, reward = self.get_next(curr_state, action)
				self.update_q(curr_state, next_state, action, reward)
				curr_state = next_state
				total_reward += reward

			rewards.append(total_reward)
			if self.action_policy == 'softmax':
				self.beta = self.beta + self.beta_inc
		return rewards


	def get_path(self):
		print "The path from start state to the goal is:"
		curr_state = self.s
		curr_step =0
		self.epsilon = 0
		path = ""
		path_obtained = False
		while(curr_step<self.max_steps):
			if(curr_state==self.g):
				print path + str(curr_state)
				path_obtained = True
				return
			action = self.select_action_epsilon_greedy(curr_state)
			next_state, reward = self.get_next(curr_state, action)
			path = path + str(curr_state) + " " + str(action) + " --> "
			curr_state = next_state
			curr_step += 1
		if not path_obtained:
			print "No path obtained from start to goal"


	def print_lake(self):
		for i in range(self.n):
			for j in range(self.n):
				if self.r[i][j] == -1:
					print "H",
				elif [i, j] == self.s:
					print "S",
				elif [i, j] == self.g:
					print "G",
				else:
					print "F",
			print ""


def plot_rewards(rewards):
	plt.plot(range(1,len(rewards)+1), rewards)
	plt.axis([0, len(rewards)+1, -5, 2])
	plt.yticks(list(plt.yticks()[0]) + [1])
	plt.xlabel("Episode Number")
	plt.ylabel("Total rewards obtained in theat episode")
	plt.show()


def main():
	n = int(sys.argv[1])
	m = int(sys.argv[2])

	fl = frozen_lake(n,m)
	fl.print_lake()
	rewards = fl.q_learn()
	fl.get_path()
	plot_rewards(rewards)


if __name__ == "__main__":
	main()

