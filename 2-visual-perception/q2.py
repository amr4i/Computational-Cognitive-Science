import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

MIN_Y = 2
MAX_Y = 48
MIN_X = 2
MAX_X = 48


# create a image to run feature search on
# having N-1 objects of one color, and one object of the other color
# all having the same shape
def create_feature_search_image_color_changed(num_objects, shape):
	objects = []
	num_objects_created = 0
	while num_objects_created < num_objects:
		# select random locations for the objects
		i = random.choice(range(MIN_X, MAX_X, 4))
		j = random.choice(range(MIN_Y, MAX_Y, 4))
		if not [i,j] in objects:
			objects.append([i,j])
			num_objects_created += 1

	X = np.array(objects)

	# select the color and odd_color choice randomly
	color = random.choice(['red', 'blue'])
	odd_color = 'red'
	if color == 'red':
		odd_color = 'blue'
	Y = [color for i in range(num_objects)]
	Y[random.choice(range(num_objects))] = odd_color
	
	fig, ax = plt.subplots(figsize=(4,4), dpi=300, frameon=False)
	# for removing all border from the around the axes
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
	# for removing all axis information 
	plt.tick_params(axis='x',which='both',bottom=False,      
    	top=False,labelbottom=False) 
	plt.tick_params(axis='y',which='both',right=False,      
    	left=False,labelleft=False) 
	# scatter plot to plot all points
	ax.scatter(X[:, 0], X[:, 1], s = 20, color = Y[:], marker=shape)

	ax.axis([MIN_Y-2, MAX_Y+2, MIN_X-2, MAX_X+2])
	# ax.axis('off')
	ax.set_aspect('equal')
	ax.margins(0)
	ax.tick_params(which='both', direction='in')

	# save the image
	img_file_name = "feature_search_image_"+str(num_objects)+".png"
	fig.savefig("images/"+img_file_name, bbox_inches='tight', pad_inches=0)
	return img_file_name


# create a image to run feature search on
# having N-1 objects of one shape, and one object of the other shape
# all having the same color
def create_feature_search_image_shape_changed(num_objects):
	objects = []
	num_objects_created = 0
	while num_objects_created < num_objects:
		# select random locations for the objects
		i = random.choice(range(MIN_X, MAX_X, 4))
		j = random.choice(range(MIN_Y, MAX_Y, 4))
		if not [i,j] in objects:
			objects.append([i,j])
			num_objects_created += 1

	X = np.array(objects)
	
	# select the color choice randomly
	color = random.choice(['red', 'blue'])
	
	# select the shape and odd_shape choice randomly
	shape = random.choice(['^', 's'])
	odd_shape = '^'
	if shape == '^':
		odd_shape = 's'
	Y = [color for i in range(num_objects)]
	
	fig, ax = plt.subplots(figsize=(4,4), dpi=300, frameon=False)
	# for removing all border from the around the axes
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
	
	# for removing all axis information 
	plt.tick_params(axis='x',which='both',bottom=False,      
    	top=False,labelbottom=False) 
	plt.tick_params(axis='y',which='both',right=False,      
    	left=False,labelleft=False) 

	# scatter plot to plot all points
	ax.scatter(X[1:, 0], X[1:, 1], s = 20, color = Y[1:], marker=shape)
	ax.scatter(X[0,0], X[0,1], s=20, color=color, marker=odd_shape)

	ax.axis([MIN_Y-2, MAX_Y+2, MIN_X-2, MAX_X+2])
	# ax.axis('off')
	ax.set_aspect('equal')
	ax.margins(0)
	ax.tick_params(which='both', direction='in')

	# save the image
	img_file_name = "feature_search_image_"+str(num_objects)+".png"
	fig.savefig("images/"+img_file_name, bbox_inches='tight', pad_inches=0)
	return img_file_name


# function to create feature search image
def create_feature_search_image(num_objects):
	# decide which feature to vary randomly
	choice = random.choice(['shape','color'])

	if choice == 'shape':
		return create_feature_search_image_shape_changed(num_objects)
	else:
		# decide the shape randomly, for the color variation case
		shape_choice = random.choice(['^', 's'])
		return create_feature_search_image_color_changed(num_objects, shape_choice)


# function to create conjunciton search image
def create_conjunction_search_image(num_objects):
	objects = []
	num_objects_created = 0
	while num_objects_created < num_objects:
		# select random locations for the objects
		i = random.choice(range(MIN_X, MAX_X, 4))
		j = random.choice(range(MIN_Y, MAX_Y, 4))
		if not [i,j] in objects:
			objects.append([i,j])
			num_objects_created += 1

	X = np.array(objects)
	# randomly select the color and shape for the odd-one-out object
	odd_color = random.choice(['red', 'blue'])
	odd_shape = random.choice(['^', 's'])
	shape = '^'
	if odd_shape == '^':
		shape = 's'
	color = 'red'
	if odd_color == 'red':
		color = 'blue'
	Y1 = [odd_color for i in range((num_objects+1)/2)]
	Y2 = [color for i in range(num_objects/2)]
	Y = Y1+Y2
	
	fig, ax = plt.subplots(figsize=(4,4), dpi=300, frameon=False)
	
	# for removing all border from the around the axes
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

	# for removing all axis information 
	plt.tick_params(axis='x',which='both',bottom=False,      
    	top=False,labelbottom=False) 
	plt.tick_params(axis='y',which='both',right=False,      
    	left=False,labelleft=False) 
	
	# scatter plots to plot all points
	ax.scatter(X[1:(num_objects+1)/2,0], X[1:(num_objects+1)/2,1], s=20, color=Y[1:(num_objects+1)/2], marker=shape)
	ax.scatter(X[(num_objects+1)/2:,0], X[(num_objects+1)/2:,1], s=20, color=Y[(num_objects+1)/2:], marker=odd_shape)
	ax.scatter(X[0,0], X[0,1], s=20, color=odd_color, marker=odd_shape)

	ax.axis([MIN_Y-2, MAX_Y+2, MIN_X-2, MAX_X+2])
	# ax.axis('off')
	ax.set_aspect('equal')
	ax.margins(0)
	ax.tick_params(which='both', direction='in')

	# save the image
	img_file_name = "conjunction_search_image_"+str(num_objects)+".png"
	fig.savefig("images/"+img_file_name, bbox_inches='tight', pad_inches=0)
	return img_file_name


if __name__ == "__main__":
	num_objects = int(sys.argv[1])
	paradigm = sys.argv[2]
	if paradigm == 'f':
		create_feature_search_image(num_objects)
	elif paradigm == 'c':
		create_conjunction_search_image(num_objects)
	else:
		print("Invalid Paradigm. Enter f or c. Exiting!")
		exit(1)