from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import qtpy
import sys
pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(0,1):
	ct, color = CTreader.read_dirty(i, r=(1000,1020))
	circle_dict  = CTreader.find_tubes(color[10])

	#CTreader.view(ct) 

	if circle_dict['labelled_img'] is not None:
		pass
		#cv2.imshow('output', circle_dict['labelled_img'])
		#cv2.waitKey()

cropped_cts = CTreader.crop(color, circle_dict['circles'])

cv2.imshow('output', cropped_cts[0][0])
cv2.waitKey()

