from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
pd.set_option('display.max_rows', None)

CTreader = CTreader()

#CTreader.view(ct)
for i in range(12):
	ct, color = CTreader.read_dirty(i, r=(950,1000))
	circles = CTreader.find_tubes(color[30])

	if circles.any():
		cv2.imshow('output', circles)
		cv2.waitKey()


#Loop over circle and change every value 
#(thresh etc) until you get 8 tubes detected



'''
crop circles to save as single fish
# given x,y are circle center and r is radius
rectX = (x - r) 
rectY = (y - r)
crop_img = self.img[y:(y+2*r), x:(x+2*r)]
'''