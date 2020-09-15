import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import correlate
import cv2
import random
import gc

def scale(X, x_min, x_max):
	# https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def find_min_value_coords(x):
	# https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
	indices = np.where(x == x.min())
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

def cc(stack, template):
	'''
	Cross correlate a ct scan against a template and return center

	Based on this paper 
	Leydon, P., O'Connell, M., Green, D., & Curran, K. (2019). 
	Cross-correlation template matching for liver localisation in computed tomography. 
	IMVIP 2019: Irish Machine Vision & Image Processing, August 28-30. 
	doi:10.21427/8fgf-y086
	'''

	x, y, z = ctreader.get_max_projections(stack) # get max projections using np.max(stack, axis=0)
	x = scale(x, -1, 1) # Rescale images to intensities of -1 to 1
	y = scale(y, -1, 1)

	# Done once on x and once on y
	xt, yt, zt = ctreader.get_max_projections(template)
	xt = scale(xt, -1, 1)
	yt = scale(yt, -1, 1)

	# plt.imshow(x); plt.show()
	# plt.imshow(xt); plt.show()
	# plt.imshow(y); plt.show()
	# plt.imshow(yt); plt.show()

	# Scipy correlate
	cx = correlate(x, xt, mode='same', method='fft')
	cy = correlate(y, yt, mode='same', method='fft')

	# plt.imshow(cx, cmap='hot', interpolation='nearest'); plt.show()
	# plt.imshow(cy, cmap='hot', interpolation='nearest'); plt.show()
	print(cx.shape, cy.shape)

	centerX = find_min_value_coords(cx)
	centerY = find_min_value_coords(cy)
	print(centerX, centerY)

	# Find the center by using correlation through axial and ventral views
	# and average y value between both (Should be relatively close)
	center = [centerY[0], centerX[0], centerX[1]]
	otolith = crop_around_center(stack, center, roiSize = 100)
	print(otolith.shape)
	ctreader.view(otolith)
	return center


if __name__ == "__main__":
	_list = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]
	ctreader = ctfishpy.CTreader()
	templatePath = './Data/Labels/CC/otolith_template_10.hdf5'
	template = ctreader.read_label(templatePath, manual=False)

	# template = crop_around_center(template, roiSize=200)
	# ctreader.view(template)

	for fish in _list:
		ct, stack_metadata = ctreader.read(fish, align=True)
		
		center = cc(ct, template)
		print(center)
	