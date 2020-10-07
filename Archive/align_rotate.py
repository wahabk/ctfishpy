import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from scipy import ndimage
from pathlib2 import Path
import json
random.seed(a = 111)

ctreader = ctfishpy.CTreader()
lumpfish = ctfishpy.Lumpfish()

dataset = Path('/home/wahab/Data/HDD/uCT/low_res_clean/')
nums = [int(path.stem) for path in dataset.iterdir() if path.is_dir()]
nums.sort()
projections = Path('Data/projections/x/')
with open('ctfishpy/angles.json', 'r') as fp:
	angles = json.load(fp)

projections = Path('Data/projections/x/')
made = [int(path.stem) for path in projections.iterdir() if path.is_file() and path.suffix == '.png']
made.sort()
nums = [x for x in nums if x not in made]
nums.sort()
print(nums)

for fish in nums:
	stack, metadata = ctreader.read(fish, align=False)
	print(fish)
	z, x, y = ctreader.make_max_projections(stack)
	ret, thresh = cv2.threshold(z, 150, 255, cv2.THRESH_BINARY) #| cv2.THRESH_OTSU)
	
	y, x = ndimage.measurements.center_of_mass(thresh)
	center = [x,y]
	int_center = (int(x),int(y))
	print(center)
 
	cv2.circle(thresh, int_center, 10, (0, 0, 255), 20)
	angle = ctreader.spin(z, center=center)

	metadata['angle'] = angle
	metadata['center'] = center
	print(angle)
	lumpfish.write_metadata(fish, metadata)
	angles[str(fish)] = angle

with open('ctfishpy/angles.json', 'r') as fp:
	json.dump(angles, fp)