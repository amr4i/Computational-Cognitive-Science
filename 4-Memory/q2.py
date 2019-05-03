import numpy as np
from tqdm import tqdm
from drawFromADist import drawFromADist

def main(verbose):
	# the temporal context model assumes that the past becomes increasingly
	# dissimilar to the future, so that memories become harder to retrieve the
	# farther away in the past they are


	N_WORLD_FEATURES = 5
	N_ITEMS = 10
	ENCODING_TIME = 500
	TEST_TIME = 20

	# we are going to model the world as a set of N continuous-valued features.
	# we will model observations of states of the world as samples from N
	# Gaussians with time-varying means and fixed variance. For simplicity,
	# assume that agents change nothing in the world.

	# first fix the presentation schedule; I'm assuming its random


	def get_optimal_schedule():
		# this schedule make the maximum possible number of jumps of maximum size, so that the median is maximized.
		num_breaks = N_ITEMS-1
		small_breaks = int((num_breaks-1)/2)
		large_breaks = int(num_breaks/2) + 1
		large_break = (1.0*(ENCODING_TIME-(small_breaks+1))/large_breaks)
		schedule = [1]
		curr = 1
		'''
		# all small breaks at start
		for i in range(small_breaks):
			curr += 1
			schedule.append(curr)
		for i in range(large_breaks):
			curr += large_break
			schedule.append(curr)
		# this leads to same schedule loss but poor success = 5.933 
		'''

		'''
		# intertwined
		while(small_breaks+large_breaks != 0):
			if large_breaks > 0:
				curr += large_break
				schedule.append(curr)
				large_breaks -= 1
			if small_breaks > 0:
				curr+=1
				schedule.append(curr)
				small_breaks -= 1
		# The intertwining of jumps of different sizes also improves the success = 7.761 
		'''
	
		# '''
		# all large breaks at start
		for i in range(large_breaks):
			curr += large_break
			schedule.append(curr)
		for i in range(small_breaks):
			curr += 1
			schedule.append(curr)
		# This leads to a furthur minor improvement in the success = 7.836
		# '''	

		return np.stack([np.array(schedule), np.arange(1,N_ITEMS+1)], axis=1) 


	# function to get the desired encoding schedule depending on users choice
	def get_encoding_schedule(choice):
		random_schedule = np.stack([np.sort(np.round(np.random.rand(1,N_ITEMS)*ENCODING_TIME))[0], np.arange(1,N_ITEMS+1)], axis=1)
		fixed_schedule_trivial = np.stack([np.arange(1,N_ITEMS+1), np.arange(1,N_ITEMS+1)], axis=1)
		fixed_schedule_equispaced = np.stack([np.round(np.linspace(1,ENCODING_TIME,N_ITEMS)), np.arange(1,N_ITEMS+1)], axis=1)
		fixed_schedule_custom = np.array([[ 79,   1],
		 [149,   2],
		 [235,   3],
		 [243,   4],
		 [246,   5],
		 [266,   6],
		 [374,   7],
		 [398,   8],
		 [398,   9],
		 [427,  10]])
		# This is the optimal one that I propose to minimise the schedule load
		# i.e. maximise the median of diff of the schedule times
		fixed_schedule_hypothesis_optimal = get_optimal_schedule()
		if choice == 'r':
			return random_schedule
		elif choice == 'c':
			return fixed_schedule
		elif choice == 't':
			return fixed_schedule_trivial
		elif choice == 'e':
			return fixed_schedule_equispaced
		elif choice == 'o':
			return fixed_schedule_hypothesis_optimal
		else:
			print("Incorrect choice for schedule! Terminating!")
			exit(1)


	'''
	# To check for the scale to which the minimum schedule load drops

	min_sl = 1000
	for i in tqdm(range(50000)):
		schedule_choice = 'r'
		schedule = get_encoding_schedule(schedule_choice)
		# variable important for parts 2,3 of assignment
		schedule_load = ENCODING_TIME/np.median(np.diff(schedule[:,0]))   
		if schedule_load < min_sl:
			min_sl = schedule_load	
			min_s = schedule
	print(min_s)            
	schedule = min_s
	print("Minimum Schedule Load: ", min_sl)
	'''

	# We got minimum schedule loss of 6.4 from 50000 random samples
	# our hypothesised optimum gives schedule loss = 5.05 which is <6.4
	# which somewhat confirms our proposal that this is the optimal one. 

	# '''
	schedule_choice = 'o'
	schedule = get_encoding_schedule(schedule_choice)
	if verbose:
		print("Encoding Schedule: ", schedule)
	# variable important for parts 2,3 of assignment
	schedule_load = ENCODING_TIME/np.median(np.diff(schedule[:,0]))   
	if verbose:
		print("Schedule Load: ", schedule_load)
	# '''

	encoding = np.zeros((N_ITEMS,N_WORLD_FEATURES+1))


	world_m = [1, 2, 1, 2, 3];              # can generate randomly for yourself
	world_var = 1;
	delta = 0.05;                       # what does this parameter affect?
	beta_param = 0.001;                 # what does this parameter affect?
	m = 0;



	# Delta being sampled from a minture of two Gaussians
	# the mixture of gaussians is fixed and known at retrieval step as well
	def get_delta():
		gmm_weights = [0.7, 0.3]
		# small gaussian has index 0, large gaussian has index 1
		mu = [0.01, 0.25]
		var = [0.01, 0.05]

		gaussian_id = np.where(np.random.multinomial(1,gmm_weights))[0][0]
		delta = np.random.normal(mu[gaussian_id], var[gaussian_id])
		return delta



	world = np.ones(5)/np.sqrt(5)

	# simulating encoding

	for time in range(1,ENCODING_TIME+1):

		delta = get_delta()
		world_m = np.add(world_m,delta);
		world_u = np.random.normal(world_m, world_var);
		dot_prod = np.dot(world, world_u)
		rho = np.sqrt(1+pow(beta_param,2)*(pow(dot_prod,2)-1)) - beta_param*(dot_prod)
		world = rho*world + beta_param*world_u

		# any item I want to encode in memory, I encode in association with the
		# state of the world at that time.
		if m < N_ITEMS:

			if time == schedule[m,0]:

				# encode into the encoding vector
				encoding[m,:] = np.concatenate((world, [m]))                                                
				m =  m + 1;




	 
	# simulating retrieval using SAM, but with a bijective image-item mapping
	out = []
	while time < (ENCODING_TIME+TEST_TIME) :
	# the state of the world is the retrieval cue
		# model world evolution
		delta = get_delta()
		world_m = np.add(world_m,delta);
		world_u = np.random.normal(world_m, world_var);
		dot_prod = np.dot(world, world_u)
		rho = np.sqrt(1+pow(beta_param,2)*(pow(dot_prod,2)-1)) - beta_param*(dot_prod)
		world = rho*world + beta_param*world_u
		
		# initialize stregth of association vector
		soa = np.zeros(N_ITEMS)
		for m in range(0,N_ITEMS):
			# retrieve all relevant contexts/states for this item
			all_contexts =  encoding[np.where(encoding[:,N_WORLD_FEATURES]==m)][:,0:N_WORLD_FEATURES]
			# get total association strength from all these contexts/states
			soa[m] = np.sum(np.dot(all_contexts, world))

		# normalize 
		out.append(np.where(drawFromADist(soa)))
		time = time + 1       

	out = np.array(out)
	success = len(np.unique(out));  
	if verbose:
		print("Success:", success)                                                
	# success is number of unique retrievals
	return success

	# humans can retrieve about 7 items effectively from memory. get this model
	# to behave like humans





if __name__ == "__main__":
	
	estimate_avg = True
	# estimate_avg = False
	if estimate_avg:
		# To get the average success over 1000 runs on the proposed optimal
		s = []
		for i in tqdm(range(1000)):
			s.append(main(verbose=False))
		print("Average Success:", np.mean(s))
		# The average success on 1000 runs using the optimal hypothesised schedule is ~7.8
		# and the schedule loss on that is 5.05
	else:
		# single run
		main(True)