import ctfishpy
from ctfishpy.viewer.circle_order_labeller import circle_order_labeller
from ctfishpy.viewer.tubeDetector import detectTubes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
from pathlib2 import Path


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()

	path = Path('/home/wahab/Data/HDD/uCT/low_res/EK_208_215')

	#circle detect
	#order labeller

	ct = lump.read_tiff(path, r=(0,5), scale = 40)
	color =  np.array([np.stack((img,)*3, axis=-1) for img in ct.copy()])

	# ctreader.view(ct)

	circle_dict = detectTubes(color)
	print(circle_dict['labelled_img'].shape)
	ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_img'], circle_dict['circles'])
	print(ordered_circles)
	print(numbered.shape)

	# TODO remember to save crop
	ctreader.view(numbered)
	ct = lump.read_tiff(path, r=(0,50), scale = 100)
	croppedCTs = lump.crop(ct, ordered_circles, scale=[40,40])
	[print(c.shape) for c in croppedCTs]
	
		