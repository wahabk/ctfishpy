from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.mainwindowcircle import detectTubes
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
import csv

pd.set_option('display.max_rows', None)

def saveCrop(ordered_circles, metadata):
	crop_data = {
		'ordered_circles' 	: ordered_circles.tolist(),
		'scale'				: metadata['scale'],
		'path'				: metadata['path']
	}
	jsonpath = metadata['path']+'/crop_data.json'
	with open(jsonpath, 'w') as o:
		json.dump(crop_data, o)

def readCrop(number):
	files = pd.read_csv('../../Data/HDD/uCT/filenames_low_res.csv', header = None)
	files = files.values.tolist()

	crop_path = '../../Data/HDD/uCT/low_res/'+files[number][0]+'/crop_data.json'
	print(crop_path)
	crop_data = pd.read_csv(crop_path) # READ JSON
	crop_data = crop_data.values.tolist()
	print(crop_data)

CTreader = CTreader()
master = CTreader.mastersheet()

'''
for i in range(0,1):
	ct, stack_metadata = CTreader.read_dirty(i, r=(900,1100), scale = 40)
	circle_dict = CTreader.find_tubes(ct)
circle_dict = detectTubes(ct)
ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
CTreader.view(numbered)
saveCrop(ordered_circles, stack_metadata)

'''




readCrop(0)

#add buttons and sequence of detect circles then label
