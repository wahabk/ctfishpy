from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(0,1):
	ct, color = CTreader.read_dirty(i, r=(18,1900), scale = 20)
	circle_dict  = CTreader.find_tubes(color, dp = 1.35, 
		minDistance = 20)

	#CTreader.view(ct) 

	if circle_dict['labelled_img'] is not None:
		continue
		cv2.imshow('output', circle_dict['labelled_img'])
		cv2.waitKey()

ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
CTreader.view(numbered)

cropped_cts = CTreader.crop(ct, ordered_circles)

for c in cropped_cts:
	c = np.mean(np.mean(c, axis = 1), axis = 1)

	plt.bar(np.arange(c.shape[0]), c, color='green')
	plt.xlabel("CT slice")
	plt.ylabel("Pixel intensity")
	plt.title("Average pixel intensity of each slice")
	plt.show()
	plt.close()



