import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from scipy import ndimage

def crop3D(stack, crop):
	return stack[crop[0,0]: crop[0,1], crop[1,0]: crop[1,1], crop[2,0]: crop[2,1]]

def makeTemplate(labelsList, scanList):
	'''
	Make template by:
	1. find lagenal centre of mass of labels 
	1. crop 256x256x256 around COM
	2. use crop from original scan (only reading necessary slices) 
	3. Average multiple ROI crops 
	'''
	shape = labelsList.shape
	cropList = []

	for label in labelsList:
		y, x, z = ndimage.measurements.center_of_mass(label)
		find_roi_bounds = lambda x: [int(x - 125.5), int(x + 125.5)]
		
		'''
		using lambda to make this shape 3d crop array
		and make cc easier:

		crop = [[x1, x2],
				[y1, y2],
				[z1, z2]]
		''' 

		crop = [find_roi_bounds(x),
				find_roi_bounds(y),
				find_roi_bounds(z)]
		cropList.append(crop)

	masterList = zip(scanList, cropList)
	for ct, crop in scanList:
		otolith = crop3D(ct, crop)
		ctreader = ctfishpy.CTreader()
		ctreader.view(otolith)
		otoliths.append(otolith)

	template = np.average(otoliths, 3)
	return template


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lumpfish = ctfishpy.Lumpfish()
	
	labelsList = []
	scanList = []
	_list = [76]#, 81, 85, 88, 218, 222, 236, 298, 425]
	for fish in _list:
		ct, metadata = ctreader.read(fish)
		label = ctreader.read_label(f'home/wahab/Data/HDD/uCT/Labels/Otolith1/076.h5')
		labelsList.append(ct)
		scanList.append(label)

	template = makeTemplate(labelsList, scanList)
	lumpfish.write_label('Data/Labels/CC/otolith_template.hdf5', template)

#/home/wahab/Data/HDD/uCT/Labels/Otolith1