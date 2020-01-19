from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
pd.set_option('display.max_rows', None)

CTreader = CTreader()

for i in range(0,1):
	ct, color = CTreader.read_dirty(i, r=(950,1000))
	output, circles  = CTreader.find_tubes(color[30])

	if output.any():
		cv2.imshow('output', output)
		cv2.waitKey()
		print(circles.shape[0]) # number of circles detected

def crop(ct, circles):
	CTs = []
	for x, y, r in circles:
		c = []
		for slice_ in ct:
			rectX = (x - r) 
			rectY = (y - r)
			cropped_slice = slice_[rectY:(y+2*r), rectX:(x+2*r)]
			c.append(cropped_slice)
		CTs.append(c)

	return CTs

cropped_cts = crop(ct, circles)
cv2.imshow('output', cropped_cts[0][0])
cv2.waitKey()

'''
crop circles to save as single fish
# given x,y are circle center and r is radius
rectX = (x - r) 
rectY = (y - r)
crop_img = self.img[y:(y+2*r), x:(x+2*r)]
'''
