import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from scipy import ndimage
from pathlib2 import Path
random.seed(a = 111)

ctreader = ctfishpy.CTreader()
lumpfish = ctfishpy.Lumpfish()

dataset = Path('/home/wahab/Data/HDD/uCT/low_res_clean/')
nums = [int(path.stem) for path in dataset.iterdir() if path.is_dir()]
nums.sort()
projections = Path('Data/projections/x/')


for fish in nums:
	metadata = ctreader.read_metadata(fish)
	projection = cv2.imread(f'Data/projections/x/{fish}.png')
	print(fish)

	ret, thresh = cv2.threshold(projection, 150, 255, cv2.THRESH_BINARY) #| cv2.THRESH_OTSU)
	
	y, x, z = ndimage.measurements.center_of_mass(thresh)
	center = [x,y]
	int_center = (int(x),int(y))
	print(center)

	cv2.circle(thresh, int_center, 10, (0, 0, 255), 20)
	angle = ctreader.spin(projection, center=center)

	metadata['angle'] = angle
	metadata['center'] = center
	print(angle)
	#lumpfish.write_metadata(fish, metadata)
