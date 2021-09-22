import ctfishpy
from ctfishpy.viewer.circle_order_labeller import circle_order_labeller
from ctfishpy.viewer.tubeDetector import detectTubes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
from pathlib2 import Path
import gc

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()

	# path = Path('/media/wahab/SeagateExp/Data/uCT/qiao/QT_020-023_[tifs]')
	path = Path('/home/wahab/Data/HDD/uCT/qiao/QT_020-023_[tifs]')
	name = path.stem

	ct = lump.read_tiff(path, r=(000,100), scale = 40)
	color =  np.array([np.stack((img,)*3, axis=-1) for img in ct.copy()]) # convert to color

	circle_dict = detectTubes(color)

	ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_img'], circle_dict['circles'])

	print(ordered_circles)
	# TODO remember to save crop
	# read in full scaleNone

	#save ram
	del ct
	del color
	gc.collect()

	ct = lump.read_tiff(path, r=None, scale = 100)
	croppedCTs = lump.crop(ct, ordered_circles, scale=[40,100])
	[print(cropped.shape) for cropped in croppedCTs]

	for i,cropped in enumerate(croppedCTs):
		print(cropped.shape)
		
		projections = ctreader.make_max_projections(cropped)

		angle, center = ctreader.spin(projections[0])
		print(angle)
		print(cropped.shape)
		cropped = ctreader.rotate_array(cropped, angle, center)
		print(cropped.shape)

		# save projections
		z, y, x = ctreader.make_max_projections(cropped)
		projections = path.parent / f'{name}_{i+1}' / 'projections'
		projections.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(projections / f'x_{i+1}.png'), x)
		cv2.imwrite(str(projections / f'y_{i+1}.png'), y)
		cv2.imwrite(str(projections / f'z_{i+1}.png'), z)

		lump.write_tif(path.parent, f'{name}_{i+1}', cropped)
		