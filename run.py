from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.mainwindowcircle import mainwin
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(0,1):
	ct, color = CTreader.read_dirty(i, r=(1000,1200), scale = 40)
	circle_dict  = CTreader.find_tubes(color)

	#CTreader.view(ct) 

	if circle_dict['labelled_img'] is not None:
		continue
		cv2.imshow('output', circle_dict['labelled_img'])
		cv2.waitKey()

#ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
#CTreader.view(numbered)

mainwin(color)

#add buttons and sequence oi detect circles then label
