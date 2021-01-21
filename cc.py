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
import json

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

def cc(n, template, thresh, roiSize):
	'''
	Cross correlate a ct scan against a template and return center
	
	Based on this paper 
	Leydon, P., O'Connell, M., Green, D., & Curran, K. (2019). 
	Cross-correlation template matching for liver localisation in computed tomography. 
	IMVIP 2019: Irish Machine Vision & Image Processing, August 28-30. 
	doi:10.21427/8fgf-y086
	'''
	
	ctreader = ctfishpy.CTreader()
	projections = ctreader.read_max_projections(n)
	projections = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in projections]
	projections = [ctreader.thresh_img(i, thresh, False) for i in projections]
	template = ctreader.crop_around_center3d(template, roiSize=roiSize)
	template_projections = ctreader.make_max_projections(template)


	_min = 1
	_max = -1

	# Rescale images and template to intensities of 1 to -1
	projections = [scale(i, _min, _max) for i in projections]
	template_projections = [scale(i, _min, _max) for i in template_projections]
	
	z, y, x = projections
	zt, yt, xt = template_projections

	# Scipy correlate in x and y projections
	cz = correlate(z, zt, mode='same', method='fft')
	cy = correlate(y, yt, mode='same', method='fft')
	cx = correlate(x, xt, mode='same', method='fft')

	# Find coordinates of the peak of cross correlates
	centerZ = find_max_value_coords(cz)
	centerY = find_max_value_coords(cy)
	centerX = find_max_value_coords(cx)
	center=[int((centerX[0] + centerY[0])/2),
			int((centerY[1] + centerZ[0])/2),
			int((centerX[1] + centerZ[1])/2)]
	


	manualcenterspath = ctreader.dataset_path / 'cc_centres.json'
	with open(manualcenterspath, "r") as fp:
		mancenters = json.load(fp)
	mancenter = mancenters[str(n)]

	#find difference in centre for each one to find how certain it is of the center
	x_diff = (center[2] - mancenter[2])**2
	y_diff = (center[1] - mancenter[1])**2
	z_diff = (center[0] - mancenter[0])**2

	# Find if any of the values are zero and if so increment error c by 1000 to make obvious

	square_error = x_diff + y_diff + z_diff
	error = int(math.sqrt(square_error))
	
	# Find the center by using correlation through x and y views

	cmap = 'Spectral'
	plt.imsave('output/cc_corr_x.png', cx, cmap=cmap)
	plt.imsave('output/cc_corr_y.png', cy, cmap=cmap)
	plt.imsave('output/cc_corr_z.png', cz, cmap=cmap)

	return center, error, mancenter


if __name__ == "__main__":
	_list = [218, 222, 236, 40, 425,81,85,88]
	roiSize = 150
	thresh = 100

	ctreader = ctfishpy.CTreader()
	template = ctreader.read_label('Otoliths', 0)

	projectionspath = ctreader.dataset_path / 'projections/x/'

	# made = [path.stem for path in projectionspath.iterdir() if path.is_file() and path.suffix == '.png']
	# made = [int(name.replace('x_', '')) for name in made]
	# made.sort()

	errors = []
	for n in _list:
		ct, metadata = ctreader.read(n, align = True)
		z,y,x = ctreader.read_max_projections(n)
		center, error, mancenter  = cc(n, template, thresh, roiSize)
		
		# center.pop(0)
		otolith = ctreader.crop_around_center3d(ct, roiSize, center)
		print(f'center {center} mancenter {mancenter} ct shape {ct.shape} otolith shape {otolith.shape} error {error} ')
		ctreader.view(otolith)
		# errors.append(int(error))
	# errors = np.array(errors)
	# np.savetxt('output/cc_errors.csv', errors)
