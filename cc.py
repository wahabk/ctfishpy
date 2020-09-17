import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import correlate
import cv2
import random
import gc
import itertools
from pathlib2 import Path
import math

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

def crop_around_center3d(array, center=None, roiSize=50):
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

def crop_around_center2d(array, center=None, roiSize=100):
	l = int(roiSize/2)
	if center:
		x, y  = center
		array = array[x-l:x+l, y-l:y+l]
	else:
		t = int(array.shape[0]/2)
		center = [t, t]
		x, y  = center
		array = array[x-l:x+l, y-l:y+l]
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
	projections = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in projections]
	projections = [ctreader.thresh_img(i, thresh, True) for i in projections]
	template = crop_around_center3d(template, roiSize=roiSize)
	template_projections = ctreader.get_max_projections(template)

	_min = 1
	_max = -1

	# Rescale images and template to intensities of 1 to 1
	projections = [scale(i, _min, _max) for i in projections]
	template_projections = [scale(i, _min, _max) for i in template_projections]
	
	x, y, z = projections
	xt, yt, zt = template_projections

	# Scipy correlate in x and y projections
	cx = correlate(x, xt, mode='same', method='fft')
	cy = correlate(y, yt, mode='same', method='fft')
	cz = correlate(z, zt, mode='same', method='fft')

	# cmap = 'Spectral'
	# plt.imsave('output/cc_corr_x.png', cx, cmap=cmap)
	# plt.imsave('output/cc_corr_y.png', cy, cmap=cmap)
	# plt.imsave('output/cc_corr_z.png', cz, cmap=cmap)

	# Find coordinates of the peak of cross correlates
	centerX = find_max_value_coords(cx)
	centerY = find_max_value_coords(cy)
	centerZ = find_max_value_coords(cz)

	#find difference in centre for each one to find how certain it is of the center
	x_diff = centerX[0] - centerZ[1]
	y_diff = centerX[1] - centerY[1]
	z_diff = centerZ[0] - centerY[0]

	# Find if any of the values are zero and if so increment error c by 1000 to make obvious
	c=1
	for i in itertools.chain(centerX, centerY, centerZ):
		if i == 0:
			c =+ 1000
	square_error = x_diff**2 + y_diff**2 + c**2
	error = int(math.sqrt(square_error))
	
	# Find the center by using correlation through x and y views
	# and find y value 
	center = [centerY[0], centerX[0], centerX[1]]
	return center, error


if __name__ == "__main__":
	_list = [85, 88, 218, 222, 236, 298, 425] #40, 76, 81, 
	ctreader = ctfishpy.CTreader()
	templatePath = './Data/Labels/CC/otolith_template_10.hdf5'
	template = ctreader.read_label(templatePath, manual=False)

	projections = Path('Data/projections/x/')
	made = [int(path.stem) for path in projections.iterdir() if path.is_file() and path.suffix == '.png']
	made.sort()

	errors = []
	for fish in made:
		# ct, stack_metadata = ctreader.read(fish)
		# projections = ctreader.get_max_projections(ct)
		
		x = cv2.imread(f'Data/projections/x/{fish}.png')
		y = cv2.imread(f'Data/projections/y/{fish}.png')
		z = cv2.imread(f'Data/projections/z/{fish}.png')
		projections = [x, y, z]
		
		center, error = cc(projections, template, thresh=200, roiSize=50)
		center.pop(0)
		otolith = crop_around_center2d(x, center = center, roiSize=255)

		# print(otolith.shape)
		# cv2.imshow('', otolith)
		# cv2.waitKey(0)

		errors.append(int(error))
	errors = np.array(errors)
	np.savetxt('cc_errors.csv', errors)

	