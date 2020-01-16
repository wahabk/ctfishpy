import CTFishPy.utility as utility
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
pd.set_option('display.max_rows', None)


CTreader = CTreader()

for i in range(12):
	ct, color = CTreader.read_dirty(i, r=(1,50))
	#CTreader.view(ct)

	circles = CTreader.find_tubes(color[0])

	if circles.any():
		cv2.imshow('output', circles)
		cv2.waitKey()

		print(circles.shape)







'''
crop circles to save as single fish
# given x,y are circle center and r is radius
rectX = (x - r) 
rectY = (y - r)
crop_img = self.img[y:(y+2*r), x:(x+2*r)]
'''