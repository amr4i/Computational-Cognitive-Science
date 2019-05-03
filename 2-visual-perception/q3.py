import q2
import q1
import os
import cv2
import sys
import time
import random
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

MAX_X = q2.MAX_X
MIN_X = q2.MIN_X
MAX_Y = q2.MAX_Y
MIN_Y = q2.MIN_Y


'''
To get the feature maps for the four different features
namely: red, blue, triangle and square
'''
def get_feature_maps(img, img_file_name):
	global frame_size

	# to check whether a given pixel is red or not
	def is_red(bgr):
		if bgr[2] > 250 and bgr[0]<10 and bgr[1]<10:
			return True
		return False

	# to check whether a given pixel is blue or not
	def is_blue(bgr):
		if bgr[0] > 250 and bgr[1]<10 and bgr[2]<10:
			return True
		return False

	# assuming square shaped image
	num_div = (MAX_X-MIN_X+4)/2
	frame_size = img.shape[0]/num_div
	filters = q1.build_filters()

	# matrices to store feature maps
	tri_matrix = np.zeros((num_div, num_div))
	sq_matrix = np.zeros((num_div, num_div))
	red_matrix = np.zeros((num_div, num_div))
	blue_matrix = np.zeros((num_div, num_div))

	temp_img = img.copy()
	# feature map images
	red_fm = img.copy()
	blue_fm = img.copy()
	sq_fm = img.copy()
	tri_fm = img.copy()
	# image that holds the gabor filter output of the complete image
	new_img = np.zeros_like(img)

	# the entire image has been segmented into frames 
	# of size (frame_size x frame_size)
	# x and y are the coordinates of the top left corner of frame
	for y in range(frame_size/2, img.shape[0]-frame_size, frame_size):
		
		# draw a grid over the frames
		# cv2.line(img, (y, 0), (y, 1199), (0,0,0), 2)
		# cv2.line(img, (0, y), (1199, y), (0,0,0), 2)
		
		for x in range(frame_size/2, img.shape[0]-frame_size, frame_size):
			
			# extract that specific image frame from the image
			img_frame = temp_img[y:y+frame_size, x:x+frame_size]
			# get the gabor filter outputs for this frame
			res, feat = q1.process(img_frame, filters)
			new_img[y:y+frame_size, x:x+frame_size] = res
			# get the score for the frame having square or triangle
			sq_score, tri_score = q1.threshold(feat)

			# thresholds for decision
			sq_threshold = 4
			tri_threshold = 1e-6
			
			# get the matrix indices for this frame
			i = (y-frame_size/2)/frame_size
			j = (x-frame_size/2)/frame_size
			
			# if its a square
			if sq_score >= sq_threshold:
				sq_matrix[i][j] = 1	
				cv2.rectangle(sq_fm,(x,y),(x+frame_size, y+frame_size), (0,0,0), 2)		# yellow (0,255,255)
			
			# if its a triangle
			if tri_score >= tri_threshold:
				tri_matrix[i][j] = 1	
				cv2.rectangle(tri_fm,(x,y),(x+frame_size, y+frame_size), (0,0,0), 2)	 	# green (0,255,0)
			
			# red channel
			if is_red(img_frame[frame_size/2][frame_size/2]):
				red_matrix[i][j] = 1
				cv2.rectangle(red_fm,(x,y),(x+frame_size, y+frame_size), (0,0,0), 2)	# cyan (255,255,0)
			
			# blue channel
			if is_blue(img_frame[frame_size/2][frame_size/2]):
				blue_matrix[i][j] = 1
				cv2.rectangle(blue_fm,(x,y),(x+frame_size, y+frame_size), (0,0,0), 2)	# purple (255,0,255)

	# write out the feature maps to images
	image_path = "feature_maps/"+img_file_name.split('.')[0]
	os.makedirs(image_path)
	cv2.imwrite(image_path+'/fm_red.png',red_fm)
	cv2.imwrite(image_path+'/fm_blue.png',blue_fm)
	cv2.imwrite(image_path+'/fm_sq.png',sq_fm)
	cv2.imwrite(image_path+'/fm_tri.png',tri_fm)
	return sq_matrix, tri_matrix, red_matrix, blue_matrix


# implementation of feature search
def feature_search(sqm, trm, rm, bm):
	# converting to sparse matrices
	sqm = sparse.csr_matrix(sqm)
	trm = sparse.csr_matrix(trm)
	rm = sparse.csr_matrix(rm)
	bm = sparse.csr_matrix(bm)

	# timer started
	start_time = time.time()

	# count non_zero items for each type of object
	square_count = sqm.count_nonzero()
	tirangle_count = trm.count_nonzero()
	red_count = rm.count_nonzero()
	blue_count = bm.count_nonzero()

	# find the odd one out
	if square_count == 1:
		i,j = sqm.nonzero()
	elif tirangle_count == 1:
		i,j = trm.nonzero()
	elif red_count == 1:
		i,j = rm.nonzero()
	elif blue_count == 1:
		i,j = bm.nonzero()

	i = i[0]
	j = j[0]

	time.sleep(0.00007)
	end_time = time.time()
	# timer ends
	
	time_taken = end_time-start_time
	return j, i, time_taken


# implementation for conjunction search
def conjunction_search(sqm, trm, rm, bm):
	# converting to sparse matrices
	sqm = sparse.csr_matrix(sqm)
	trm = sparse.csr_matrix(trm)
	rm = sparse.csr_matrix(rm)
	bm = sparse.csr_matrix(bm)

	# storing indices of each type fo objects
	rt=list([])
	rs=list([])
	bt=list([])
	bs=list([])

	# timer started
	start_time = time.time()

	# count non_zero items for each type of object
	sqm_y, sqm_x = sqm.nonzero()
	trm_y, trm_x = trm.nonzero()
	rm_y, rm_x = rm.nonzero()
	bm_y, bm_x = bm.nonzero()
	sqm_indices = zip(sqm_y, sqm_x)
	trm_indices = zip(trm_y, trm_x)
	rm_indices = zip(rm_y, rm_x)
	bm_indices = zip(bm_y, bm_x)

	for index in sqm_indices:
		if index in rm_indices:
			rs.append(index)
		elif index in bm_indices:
			bs.append(index) 

	for index in trm_indices:
		if index in rm_indices:
			rt.append(index)
		elif index in bm_indices:
			bt.append(index)
	num_objects = len(rt)+len(rs)+len(bt)+len(bs)

	# simulated delay for conjunction search
	time.sleep((num_objects)*0.000005)

	end_time = time.time()
	# timer ends

	# another way to implement the conjunction search w/o sparse matrices
	'''
	# timer
	start_time = time.time()
	for i in range(sqm.shape[0]):
		for j in range(sqm.shape[1]):
			if rm[i][j]==1 and trm[i][j]==1:
				rt.append((i,j))
			if rm[i][j]==1 and sqm[i][j]==1:
				rs.append((i,j))
			if bm[i][j]==1 and trm[i][j]==1:
				bt.append((i,j))
			if bm[i][j]==1 and sqm[i][j]==1:
				bs.append((i,j))
	num_objects = len(rt)+len(rs)+len(bt)+len(bs)
	# time.sleep((num_objects**0.8)*0.0001)
	end_time = time.time()
	# timer ends
	'''

	time_taken = end_time - start_time

	# get the odd one out
	if len(rt) == 1:
		return rt[0][1], rt[0][0], time_taken
	elif len(rs) == 1:
		return rs[0][1], rs[0][0], time_taken
	elif len(bt) == 1:
		return bt[0][1], bt[0][0], time_taken
	elif len(bs) == 1:
		return bs[0][1], bs[0][0], time_taken


# to create the necessary directory structure for storing all outputs
def create_dir_structure():
	if os.path.exists("images/"):
		os.system("rm -rf ./images/")
	if os.path.exists("feature_maps/"):
		os.system("rm -rf ./feature_maps/")
	if os.path.exists("results/"):
		os.system("rm -rf ./results/")
	os.makedirs('images')
	os.makedirs('feature_maps')
	os.makedirs('results')


# implementation to simulate feature integration theory
def feature_integration_theory():
	create_dir_structure()
	num_objects = [1,5,10,15,25,30,40,50]
	# num_objects = [5, 15, 40]
	fs_time = []
	cs_time = []

	for n in num_objects:
		
		# feature search
		img_file_name = q2.create_feature_search_image(n)
		print img_file_name
		img = cv2.imread("images/"+img_file_name)
		sqm, trm, rm, bm = get_feature_maps(img, img_file_name)
		fs_x, fs_y, fs_time_taken = feature_search(sqm, trm, rm, bm)
		print_image_with_bb(img, fs_x, fs_y, "results/result_"+img_file_name)
		
		# conjunction search
		img_file_name = q2.create_conjunction_search_image(n)
		print img_file_name
		img = cv2.imread("images/"+img_file_name)
		sqm, trm, rm, bm = get_feature_maps(img, img_file_name)
		cs_x, cs_y, cs_time_taken = conjunction_search(sqm, trm, rm, bm)
		print_image_with_bb(img, cs_x, cs_y, "results/"+img_file_name)

		fs_time.append(fs_time_taken)
		cs_time.append(cs_time_taken)

	plot_results(num_objects, fs_time, cs_time)


# to plot and store the results of feature integration theory simulation
def plot_results(num_objects, fs_time, cs_time):
	fig,ax = plt.subplots()
	ax.plot(num_objects, fs_time, 'ro-', label='feature_search')
	ax.plot(num_objects, cs_time, 'bs-', label='conjunction_search')
	ax.legend()
	ax.set_ylabel('Response Time (in seconds)')
	ax.set_xlabel('Number of objects')
	plt.subplots_adjust(left=0.15)
	fig.savefig("plot.png")
	# plt.show()


# to print the bounding box on a frame in the image
def print_image_with_bb(img, x, y, filename):
	global frame_size
	x = x*frame_size + (frame_size/2)
	y = y*frame_size + (frame_size/2)
	cv2.rectangle(img,(x,y),(x+frame_size, y+frame_size), (0,0,0), 2)
	cv2.imwrite(filename, img)


if __name__ == "__main__":
	# img_name = sys.argv[1]
	# img = cv2.imread(img_name)
	# sqm, trm, rm, bm = get_feature_maps(img)
	# x, y, time_taken = feature_search(sqm, trm, rm, bm)
	# x, y, time_taken = conjunction_search(sqm, trm, rm, bm)
	# print_image_with_bb(img, x, y)
	feature_integration_theory()
