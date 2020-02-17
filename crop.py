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




CTreader = CTreader()
master = CTreader.mastersheet()
'''
#for i in range(0,1):
ct, stack_metadata = CTreader.read_dirty(0, r=(900,1100), scale = 40)
circle_dict = CTreader.find_tubes(ct)

circle_dict = detectTubes(ct)
ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
CTreader.view(numbered)
CTreader.saveCrop(ordered_circles, stack_metadata)

#import pdb; pdb.set_trace()
'''

ct, current_stack_metadata = CTreader.read_dirty(0, r=(500,1500), scale = 60)

crop_data = CTreader.readCrop(0)
scale = [crop_data['scale'], current_stack_metadata['scale']]
cropped_cts = CTreader.crop(ct, crop_data['ordered_circles'], scale = scale)
for c in cropped_cts: 
	print(c.shape)
	CTreader.view(c)

