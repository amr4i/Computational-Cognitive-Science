import sys
import copy
import numpy as np
from tqdm import tqdm
from random import shuffle

# the parameters for the model and the learning
num_functions = 5
num_variables = 3
learning_rate = 0.05
num_hidden_layers = 1
num_nodes_in_layers = [15]
max_epochs = 30000


"""
function to get all possible boolean functions 
"""
def create_all_funcs(num_variables):
	global bool_outs
	lists = []
	num_entries = pow(2,num_variables)
	for i in range(num_entries-1, -1, -1):
		_a = [0]*pow(2,i) + [1]*pow(2,i)
		_a = _a*pow(2, num_entries-1-i)
		lists.append(_a)
	bool_outs = np.array(lists).transpose()


"""
function to generate a boolean function randomly
"""
def generate_bool_func(num_variables, num):
	num_entries = pow(2,num_variables)
	bool_func = []
	lists = []
	# to obtain all possible input combinations
	for i in range(num_variables-1, -1, -1):
		_a = [0]*pow(2,i) + [1]*pow(2,i)
		_a = _a*pow(2, num_variables-1-i)
		lists.append(_a)
	_a = []
	# to randomly assign truth values for the boolean function
	for i in range(num_entries):
		if np.random.rand() > 0.5:
			_a.append(1)
		else:
			_a.append(0)
	lists.append(_a)
	bool_func = np.array(lists).transpose()
	return bool_func



"""
function to take a boolean function as input from the user
"""
def take_func_input(num_variables):
	num_entries = pow(2,num_variables)
	bool_func = []
	for i in range(num_entries):
		entry = [ int(j) for j in raw_input().split() ]
		bool_func.append(entry)
	return np.array(bool_func)	



"""
function to create a dataset of training examples from the boolean function
"""
def create_dataset(bool_func, size_multiplier):
	xtrain = []
	ytrain = []
	xpredict = []
	ypredict = []
	for i in range(size_multiplier):
		shuffle(bool_func)
		xtrain = xtrain + copy.deepcopy(bool_func)
	for a in xtrain:
		ytrain.append([a.pop()])
	xtrain = np.array(xtrain)
	ytrain = np.array(ytrain)
	shuffle(bool_func)
	xpredict = copy.deepcopy(bool_func)
	for a in xpredict:
		ypredict.append([a.pop()])	
	xpredict = np.array(xpredict)
	ypredict = np.array(ypredict)
	return xtrain, ytrain, xpredict, ypredict



"""
the definition of the neural network
"""
class ANN:
	def __init__(self, x, y, num_hidden_layers, num_neurons_in_layers, learning_rate):
		self.input = x
		self.learning_rate = learning_rate
		self.num_layers = num_hidden_layers
		self.weights = []
		self.bias = []
		self.num_neurons_in_layers = num_neurons_in_layers
		for i in range(self.num_layers):
			if i==0:
				self.weights.append(np.random.rand(self.input.shape[1], self.num_neurons_in_layers[i]))
			else:
				self.weights.append(np.random.rand(self.num_neurons_in_layers[i-1], self.num_neurons_in_layers[i]))
			self.bias.append(np.random.rand(self.num_neurons_in_layers[i]))
		self.weights.append(np.random.rand(self.num_neurons_in_layers[self.num_layers-1], 1))
		self.bias.append(np.random.rand(1))
		self.y = y
		self.output = np.zeros(self.y.shape)
		self.loss = 0
		self.layer_outputs = [0]*(self.num_layers+1)

	def forward_propagate(self):
		input_to_layer = self.input
		# print self.input
		for i in range(self.num_layers + 1):
			layer_primary_output = np.dot(input_to_layer, self.weights[i])
			layer_output = np.add(layer_primary_output, self.bias[i])
			self.layer_outputs[i] = self.activation_function(layer_output)
			input_to_layer = self.layer_outputs[i]
		self.output = input_to_layer
		

	def backpropagation(self):
		self.loss = np.sum(np.square(self.output - self.y))
		error = self.y - self.output
		for i in range(self.num_layers, -1, -1):
			der_act = self.activation_derivative(self.layer_outputs[i])
			delta = np.multiply(error,der_act)
			if i > 0:
				self.weights[i] += self.learning_rate*np.dot(self.layer_outputs[i-1].T,  delta)
				self.bias[i] += self.learning_rate*np.sum(delta, axis=0)
				error = delta.dot(self.weights[i].T)
			else:
				self.weights[i] += self.learning_rate*np.dot(self.input.T, delta)
				self.bias[i] += self.learning_rate*np.sum(delta, axis=0)

	def activation_function(self, x):
		# sigmoid activation
		return 1/(1+np.exp(-x))

	def activation_derivative(self, x):
		return np.multiply(x,np.subtract(np.ones([x.shape[0],1]),x))


	def train(self, epochs):
		print "Training neural network..."
		for i in tqdm(range(epochs)):
			self.forward_propagate()
			self.backpropagation()
		print "Total training loss: " + str(self.loss)
		print "Training complete"

	def predict(self, x, y):
		self.input = x
		self.forward_propagate()
		ypred = np.array([ [1 if i[0]>0.5 else 0] for i in self.output ])
		print "Predicted Output: "
		print ypred
		print "Actual Output: "
		print y
		if np.array_equal(ypred, y):
			print "Function learned correctly!"
			return 1
		else:
			return 0



def main():
	correctly_learned = 0
	# create_all_funcs(3)
	for i in range(num_functions):
		print("----------------------------------------------")
		print("For function number " + str(i+1))
		print("----------------------------------------------")
		
		bool_func = generate_bool_func(num_variables, i)
		print("Function:")
		print bool_func 
		# bool_func = take_func_input(num_variables)
		xtrain, ytrain, xpredict, ypredict = create_dataset(bool_func.tolist(), 1)

		neural_network = ANN(xtrain, ytrain, num_hidden_layers, num_nodes_in_layers, learning_rate)
		neural_network.train(max_epochs)
		res = neural_network.predict(xpredict, ypredict)
		correctly_learned += res

	print "Total "+str(correctly_learned)+"/"+str(num_functions)+" functions were learned perfectly! "



if __name__ == "__main__":
	main()	
