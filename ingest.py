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

	ct = lump.read_tiff(path, r=(0,5), scale = 40)
	color =  np.array([np.stack((img,)*3, axis=-1) for img in ct.copy()]) # convert to color

	circle_dict = detectTubes(color)

	ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_img'], circle_dict['circles'])


	# print(circle_dict['labelled_img'].shape)
	# print(ordered_circles)
	# print(numbered.shape)
	# TODO remember to save crop
	# read in full scale

	ct = lump.read_tiff(path, r=(0,50), scale = 100)
	croppedCTs = lump.crop(ct, ordered_circles, scale=[40,100])

	for i,cropped in enumerate(croppedCTs):
		print(cropped.shape)
		
		projections = ctreader.make_max_projections(cropped)

		angle = ctreader.spin(projections[0])
		print(angle)
		print(cropped.shape)
		cropped = ctreader.rotate_array(cropped, angle)
		print(cropped.shape)

		# save projections
		z, y, x = ctreader.make_max_projections(ct)
		projections = path.parent / f'qiao_{i+1}' / 'projections'
		projections.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(projections / f'x_{i+1}.png'), x)
		cv2.imwrite(str(projections / f'y_{i+1}.png'), y)
		cv2.imwrite(str(projections / f'z_{i+1}.png'), z)

		lump.write_tif(path.parent, f'qiao_{i+1}', cropped)

	
		