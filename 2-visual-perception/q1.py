import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d import Axes3D
from skimage.filters import gabor_kernel

MIN_Y = 4
MAX_Y = 6
MIN_X = 4
MAX_X = 6


# to create a image with a single shape, square(s) or triangle(^)
def create_shapes(shape):
	X = np.array([[5, 5]])
	Y = ['red']
	plt.figure()

	# marker = ^ (for triangle), s for square
	plt.scatter(X[:, 0], X[:, 1], s = 1070, color = Y[:], marker=shape)
	plt.axis([MIN_Y, MAX_Y, MIN_X, MAX_X])
	plt.axis('off')
	plt.savefig("test.png", bbox_inches='tight', pad_inches=0)


# build the required Gabor filters
# at orietations pi, pi/6, pi/2 and 5.pi/6
def build_filters():
	filters = []
	# the kernel parameters 
	sigma = 4.0
	lambdaa = 10.0
	frequency = 1/lambdaa
	gamma = 0.5
	psi = 0

	for theta in [ np.pi/6, np.pi/2, 5*np.pi/6, np.pi ]:
		kern = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
		filters.append(kern)
	return filters
	

# function to apply all the filters on the image
# and return the resultant image
# and the variance in the result being the feature 
def process(img, filters):
	accum = np.zeros_like(img)
	filter_features = []
	for kern in filters:
		fimg = cv2.filter2D(img, -1, kern)
		filter_features.append(np.var(fimg, axis=None))
		accum = np.maximum(accum, fimg)
	return accum, filter_features


# provide a square_score and triangle_score based on the features 
# extracted from the gabor filters
def threshold(feat):
	sq_score = (feat[1]+feat[3])/2.0 - (feat[0]+feat[2])/2.0
	tri_score = (feat[0]+feat[1]+feat[2])/3.0 - feat[3]
	return sq_score, tri_score


# function to read image 
def read_image(img_path):
	
	img = cv2.imread(img_path)

	# a blank white image for debugging 
	# img = np.ones((224,224))*255
	
	# to convert to binary image
	# ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	# cv2.imwrite('test_img.png', img)

	if img is None:
		print('Failed to load image file: '+img_path)
		sys.exit(1)

	return img



if __name__ == '__main__':

	if len(sys.argv) != 2:
		print("Please enter in format:\n python q1.py <s/t>. Exiting!")
		sys.exit(1)

	choice = sys.argv[1]
	if choice == 't':
		# to create a triangle
		create_shapes("^")
	elif choice == 's':	
		# to create a square
		create_shapes("s")
	else:
		print("Wrong Choice. Enter s or t! Exiting!")
		sys.exit(1)

	img = read_image('test.png')
	filters = build_filters()

	res, feat = process(img, filters)
	sq_score, tri_score = threshold(feat)
	print("Square Score: "+str(sq_score))
	print("Triangle Score: "+str(tri_score))	

	sq_threshold = tri_threshold = 1e-6
	if sq_score >= sq_threshold:
		print(" * Square is present!")
	if tri_score >= tri_threshold:
		print(" * Triangle is present!")

	cv2.imwrite("output.png", res)
