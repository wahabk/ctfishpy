from ctfishpy.viewer.circle_order_labeller import circle_order_labeller
from ctfishpy import CTreader, Lumpfish
import numpy as np
import cv2
import sys


if __name__ == "__main__":

	CTreader = CTreader()
	lump= Lumpfish()

	for i in range(0,1):
		ct = lump.read_dirty(i, r=(900,1100), scale = 30)
		color =  np.array([np.stack((img,)*3, axis=-1) for img in ct])
		circle_dict  = lump.find_tubes(color, slice_to_detect = 100, dp = 1.45)
		
		if circle_dict is not None:
			continue
			cv2.imshow('output', circle_dict['labelled_img'])
			cv2.waitKey()

	ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
	CTreader.view(numbered)
