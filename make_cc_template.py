import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from scipy import ndimage

def crop3D(stack, crop):
	cropped = stack[crop[0,0]: crop[0,1],
					crop[1,0]: crop[1,1],
					crop[2,0]: crop[2,1]]
	return np.array(cropped)

def check(x, x_length):
	return x
	if x < 0: 
		return 0 
	elif x > x_length:  
		return x_length

def find_roi_bounds(x, roiSize, x_length):
	x1 = check(int(x - roiSize/2), x_length)
	x2 = check(int(x + roiSize/2), x_length)
	return [x1, x2]

def makeTemplate(labelsList, scanList, roiSize):
	'''
	Make template by:
	1. find lagenal centre of mass of labels 
	1. crop 256x256x256 around COM
	2. use crop from original scan (only reading necessary slices) 
	3. Average multiple ROI crops 
	'''
	cropList = []

	print('Finding ROIs...')
	for label in labelsList:
		z, x, y = ndimage.measurements.center_of_mass(label)
		
		'''
		using lambda to make this shape 3d crop 
		array with less repitition and make cc easier:

		crop = [[z1, z2],
				[x1, x2],
				[y1, y2]]
		''' 
		labelShape = label.shape
		crop = [find_roi_bounds(z, roiSize, labelShape[0]),
				find_roi_bounds(x, roiSize, labelShape[1]),
				find_roi_bounds(y, roiSize, labelShape[2])]
		cropList.append(np.array(crop))

	otolithList = []
	for i, ct in enumerate(scanList):
		otolith = crop3D(ct, cropList[i])
		ctreader = ctfishpy.CTreader()
		
		otolith = ctreader.thresh_stack(otolith, 150)
		otolithList.append(otolith)

	otolithArray = np.array(otolithList)
	template = np.array(np.average(otolithArray, 0), dtype='uint16')
	return template


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lumpfish = ctfishpy.Lumpfish()
	templatePath = './Data/Labels/CC/otolith_template_10.hdf5'

	# labelsList = []
	# scanList = []
	# _list = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]
	# for fish in _list:
	# 	ct, metadata = ctreader.read(fish, align=True)
	# 	label = ctreader.read_label(f'../../Data/HDD/uCT/Labels/Otolith1/{fish}.h5', align=True, n=fish)

	# 	labelsList.append(label)
	# 	scanList.append(ct)

	# template = makeTemplate(labelsList, scanList, roiSize = 255)
	# #lumpfish.write_label(templatePath, template)
	template = ctreader.read_label(templatePath, manual=False)
	ctreader.view(template)


#/home/wahab/Data/HDD/uCT/Labels/Otolith1