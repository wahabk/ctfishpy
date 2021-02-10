import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from scipy import ndimage

def find_roi_bounds(x, roiSize, x_length):
	x1 = int(x - roiSize/2)
	x2 = int(x + roiSize/2)
	return [x1, x2]

def makeTemplate(samples, roiSize, thresh):
	'''
	Make template by:
	1. find centre of mass of labels 
	1. crop 256x256x256 around COM
	2. use crop from original scan (only reading necessary slices) 
	3. Average multiple ROI crops 
	'''
	
	ctreader = ctfishpy.CTreader()
	labelsList = []
	scanList = []
	for n in samples:
		ct, metadata = ctreader.read(n, align=True)
		label = ctreader.read_label('Otoliths', n, align=True, is_amira=True)
		
		labelsList.append(label)
		scanList.append(ct)

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
		crop = cropList[i]
		ct = np.array(ct, dtype = 'uint16')
		cropped = ct[crop[0,0]: crop[0,1],
					 crop[1,0]: crop[1,1],
					 crop[2,0]: crop[2,1]]
		otolith =  np.array(cropped)
		
		# otolith = ctreader.thresh_stack(otolith, thresh)
		otolithList.append(otolith)

	otolithArray = np.array(otolithList)
	template = np.average(otolithArray, axis=0)
	template = np.array(template, dtype='uint16')
	return template

if __name__ == "__main__":
	samples = [200,218,240,277,330,337,341,462,464,40,78]+[242,256,259,421,423,459,463,530,589]+[257,443,461,527,582]
	roiSize = 224
	thresh = 100
	
	ctreader = ctfishpy.CTreader()

	template = makeTemplate(samples, roiSize, thresh)
	print(template.shape, np.max(template), np.mean(template), np.min(template), template.dtype)
	ctreader.write_label(template, 'Otoliths', 0)
	ctreader.view(template)
	# template = ctreader.read_label('Otoliths', 0, align=False)



#/home/wahab/Data/HDD/uCT/Labels/Otolith1