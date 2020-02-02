from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from CTFishPy.qt import labeller
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer

if __name__ == "__main__":

	CTreader = CTreader()

	for i in range(0,1):
		ct, color = CTreader.read_dirty(i, r=(1000,1020))
		circle_dict  = CTreader.find_tubes(color[10])

		#CTreader.view(ct) 
		#print(circle_dict['circles'])
		
		if circle_dict['labelled_img'] is not None:
			continue
			cv2.imshow('output', circle_dict['labelled_img'])
			k = cv2.waitKey()

	circles = labeller(circle_dict['labelled_img'], circle_dict['circles'])


	im = circle_dict['labelled_img']
	
	k = 1
	for x, y, r in circles:
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(im, str(k), (x,y), font, 4, (255,255,255), 2, cv2.LINE_AA)
		k = k+1
	cv2.imshow('labelled circles?', im)
	cv2.waitKey()
	