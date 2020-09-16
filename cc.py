import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import correlate
import cv2
import random
import gc
import itertools

def scale(X, x_min, x_max):
	# https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
	nom = (X-X.min(axis=0))*(x_max-x_min)
	denom = X.max(axis=0) - X.min(axis=0)
	denom[denom==0] = 1
	return x_min + nom/denom

def find_max_value_coords(x):
	# https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
	indices = np.where(x == x.max())
	x_y_coords =  [indices[0][0], indices[1][0]]
	return x_y_coords

def crop_around_center(array, center=None, roiSize=50):
	l = int(roiSize/2)
	if center:
		z, x, y  = center
		array = array[z-l:z+l, x-l:x+l, y-l:y+l]
	else:
		t = int(template.shape[0]/2)
		center = [t, t, t]
		z, x, y  = center
		array = array[z-l:z+l, x-l:x+l, y-l:y+l]
	return array

def cc(projections, template, thresh, roiSize):
	'''
	Cross correlate a ct scan against a template and return center

	Based on this paper 
	Leydon, P., O'Connell, M., Green, D., & Curran, K. (2019). 
	Cross-correlation template matching for liver localisation in computed tomography. 
	IMVIP 2019: Irish Machine Vision & Image Processing, August 28-30. 
	doi:10.21427/8fgf-y086
	'''
	#projections = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in projections]
	projections = [ctreader.thresh_img(i, thresh, True) for i in projections]
	template = crop_around_center(template, roiSize=roiSize)

	x, y, z = projections
	_min = 1
	_max = -1
	# Rescale images and template to intensities of 1 to 1
	x = scale(x, _min, _max)
	y = scale(y, _min, _max)
	z = scale(z, _min, _max)

	xt, yt, zt = ctreader.get_max_projections(template)
	xt = scale(xt, _min, _max)
	yt = scale(yt, _min, _max)
	zt = scale(zt, _min, _max)

	plt.imshow(x); plt.show() 
	plt.imshow(xt); plt.show() 
	plt.imshow(y); plt.show() 
	plt.imshow(yt); plt.show() 

	# Scipy correlate in x and y projections
	cx = correlate(x, xt, mode='same', method='fft')
	cy = correlate(y, yt, mode='same', method='fft')
	cz = correlate(z, zt, mode='same', method='fft')

	# cmap = 'Spectral'
	# plt.imshow(cx, cmap=cmap, interpolation='nearest'); plt.show()
	# plt.imshow(cy, cmap=cmap, interpolation='nearest'); plt.show()

	# Find coordinates of the peak of cross correlates
	centerX = find_max_value_coords(cx)
	centerY = find_max_value_coords(cy)
	centerZ = find_max_value_coords(cz)

	#find difference in centre for each one to find how certain it is of the center
	x_diff = centerX[0] - centerY[1]
	y_diff = centerX[1] - centerY[1]
	z_diff = centerX[1] - centerY[1]

	# Find if any of the values are zero and if so increment error c
	c=0
	for i in itertools.chain(centerX, centerY, centerZ):
		if i == 0:
			c =+ 1000
	
	square_error = x_diff^2 + y_diff^2 + z_diff^2 + c^2
	print(f'Error = {square_error}')
	
	# Find the center by using correlation through x and y views
	# and find y value 
	center = [centerY[0], centerX[0], centerX[1]]
	return center


if __name__ == "__main__":
	_list = [85, 88, 218, 222, 236, 298, 425] #40, 76, 81, 
	ctreader = ctfishpy.CTreader()
	templatePath = './Data/Labels/CC/otolith_template_10.hdf5'
	template = ctreader.read_label(templatePath, manual=False)

	# TODO Make projections reader get images and align them

	for fish in _list:
		ct, stack_metadata = ctreader.read(fish)
		projections = ctreader.get_max_projections(ct)
		
		# x = cv2.imread(f'Data/projections/x/{fish}.png')
		# y = cv2.imread(f'Data/projections/y/{fish}.png')
		# z = cv2.imread(f'Data/projections/z/{fish}.png')
		# projections = [x, y, z]
		

		center = cc(projections, template, thresh=200, roiSize=50)
	