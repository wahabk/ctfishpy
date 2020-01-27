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
	ct, color = CTreader.read_dirty(i, r=(1600,2000))
	output, circles  = CTreader.find_tubes(color[0])

	CTreader.view(ct) 

	if output.any():
		cv2.imshow('output', output)
		cv2.waitKey()

cropped_cts = CTreader.crop(color, circles)

for i in range(8):
	#CTreader.view(cropped_ct)
	output, circles = None, None
	print(cropped_cts[i][0][0])
	cv2.imshow('output', cropped_cts[i][50])
	cv2.waitKey()
	output, circles = CTreader.find_tubes(cropped_cts[i][50], minRad = 0, maxRad = 50)
	if output is not None:
		cv2.imshow('output', output)
		cv2.waitKey()
	output, circles = None, None



#Rename folders using python os.rename
#

#cv2.imshow('output', cropped_cts[0][0])
#cv2.waitKey()

