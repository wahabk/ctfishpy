from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(9,10):
	ct, color = CTreader.read_dirty(i, r=(1800,2000))
	circle_dict  = CTreader.find_tubes(color[0])

	CTreader.view(ct) 

	if circle_dict['labelled_img'] is not None:
		cv2.imshow('output', circle_dict['labelled_img'])
		cv2.waitKey()

cropped_cts = CTreader.crop(color, circle_dict['circles'])

for cropped_ct in cropped_cts:
	#CTreader.view(cropped_ct)
	#print(cropped_ct[50])
	circle_dict = CTreader.find_tubes(cropped_ct[50], minRad = 0, maxRad = 50)
	if circle_dict is not None:
		cv2.imshow('output', circle_dict['labelled_img'])
		cv2.waitKey()


#Rename folders using python os.rename

#cv2.imshow('output', cropped_cts[0][0])
#cv2.waitKey()

