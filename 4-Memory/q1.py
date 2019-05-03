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

	# one of the random schedules that gave a success of 7 has been chosen as the fixed schedule
	fixed_schedule = np.array([[ 79,   1],
	 [149,   2],
	 [235,   3],
	 [243,   4],
	 [246,   5],
	 [266,   6],
	 [374,   7],
	 [398,   8],
	 [398,   9],
	 [427,  10]])

	# schedule = np.stack([np.sort(np.round(np.random.rand(1,N_ITEMS)*ENCODING_TIME))[0], np.arange(1,N_ITEMS+1)], axis=1)
	schedule = fixed_schedule
	# print(schedule)

	# variable important for parts 2,3 of assignment
	schedule_load = ENCODING_TIME/np.median(np.diff(schedule[:,0]))               

	encoding = np.zeros((N_ITEMS,N_WORLD_FEATURES+1))


	world_m = [1, 2, 1, 2, 3];              # can generate randomly for yourself
	world_var = 1;
	delta = 0.05;                       # what does this parameter affect?
	beta_param = 0.001;                 # what does this parameter affect?
	m = 0;


	world = np.ones(5)/np.sqrt(5)
	
	# simulating encoding

	for time in range(1,ENCODING_TIME+1):

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
		world_m = np.add(world_m,delta);
		world_u = np.random.normal(world_m, world_var);
		dot_prod = np.dot(world, world_u)
		rho = np.sqrt(1+pow(beta_param,2)*(pow(dot_prod,2)-1)) - beta_param*(dot_prod)
		world = rho*world + beta_param*world_u
		
		# initialize strength of association vector
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
	if(verbose):
		print("Success:", success)                                                
	# success is number of unique retrievals

	return success
	# humans can retrieve about 7 items effectively from memory. get this model
	# to behave like humans


if __name__ == "__main__":
	
	estimate_avg = True
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