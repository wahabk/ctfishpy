try: from .CTreader import CTreader
except: import CTreader
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
import skimage.filters
from sklearn.preprocessing import minmax_scale

def thresh_img(img, thresh_8, is_16bit=False):
	"""
	Threshold CT img 
	Default is 8bit thresholding but make 16_bit=True if not
	"""
	#provide threshold in 8bit since it's more intuitive then convert to 16
	thresh_16 = thresh_8 * (65535 / 255)
	if is_16bit:
		thresh = thresh_16
	if not is_16bit:
		thresh = thresh_8


	img[img<thresh] = 0
	return img

def find_max_value_coords(x):
	# https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
	indices = np.where(x == x.min())
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
	

	ctreader = CTreader()
	projections = ctreader.read_max_projections(n)
	projections = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in projections]
	projections = [thresh_img(i, thresh, is_16bit=False) for i in projections]
	template = ctreader.crop_around_center3d(template, roiSize=roiSize)
	
	template_projections = ctreader.make_max_projections(template)
	template_projections = [thresh_img(i, thresh, is_16bit=True) for i in template_projections]

	#print(template.shape, np.max(template), np.mean(template), np.min(template), template.dtype)
	
	# Renorm images and template to intensities of 1 to -1
	projections = [minmax_scale(i, feature_range=(-1,1)) for i in projections]
	template_projections = [minmax_scale(i, feature_range=(-1,1)) for i in template_projections]
	
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


	center=[int((centerX[0]+centerY[0])/2),
			int((centerZ[0]+centerX[1])/2),
			int((centerZ[1]+centerY[1])/2)]
	



	# mancenters = ctreader.manual_centers
	# mancenter = mancenters[str(n)]
	# x_diff = (center[2] - mancenter[2])**2
	# y_diff = (center[1] - mancenter[1])**2
	# z_diff = (center[0] - mancenter[0])**2
	# square_error = x_diff + y_diff + z_diff
	# error = int(math.sqrt(square_error))
	# print('CENTERS', centerZ,centerY,centerX)
	

	# cmap = 'Spectral'
	# plt.imsave('output/cc_corr_x.png', cx, cmap =cmap)
	# plt.imsave('output/cc_corr_y.png', cy, cmap =cmap)
	# plt.imsave('output/cc_corr_z.png', cz, cmap =cmap)

	return center


if __name__ == "__main__":
	_list = [385]#[459,463,530,589,257,443,461,527,582]
	#[40,200,218,240,277,330,337,341,462,464,40] # 421 and 
	roiSize = 224
	thresh = 80

	ctreader = ctfishpy.CTreader()
	template = ctreader.read_label('Otoliths', 0)
	

	# made = [path.stem for path in projectionspath.iterdir() if path.is_file() and path.suffix == '.png']
	# made = [int(name.replace('x_', '')) for name in made]
	# made.sort()

	errors = []
	for n in _list:
		ct, metadata = ctreader.read(n, align = True)
		z,y,x = ctreader.read_max_projections(n)
		center  = cc(n, template, thresh, roiSize)
		
		otolith = ctreader.crop_around_center3d(ct, roiSize, center)
		ctreader.view(otolith)
	# 	errors.append(int(error))
	# errors = np.array(errors)
	# print(errors, np.mean(errors), np.max(errors), np.min(errors)) 
	# np.savetxt('output/cc_errors.csv', errors)
