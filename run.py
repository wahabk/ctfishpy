import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from natsort import natsorted, ns
from qtpy.QtCore import QSettings
from pathlib2 import Path
from tqdm import tqdm
import tifffile as tiff
import pandas as pd
import numpy as np 
import json
import cv2
import os
import gc
random.seed(a = 111)

ctreader = ctfishpy.CTreader()

def read_dirty(file_number = None, r = None, 
	scale = 40):
	path = '../../Data/HDD/uCT/low_res/'
	
	# find all dirty scan folders and save as csv in directory
	files      = os.listdir(path)
	files      = natsorted(files, alg=ns.IGNORECASE) #sort according to names without leading zeroes
	files_df    = pd.DataFrame(files) #change to df to save as csv
	files_df.to_csv('../../Data/HDD/uCT/filenames_low_res.csv', index = False, header = False)
	
	fish_nums = []
	for f in files:
		nums = [int(i) for i in f.split('_') if i.isdigit()]
		if len(nums) == 2:
			start = nums[0]
			end = nums[1]+1
			nums = list(range(start, end))
		fish_nums.append(nums)
	fish_order_nums = fish_nums#[[files[i], fish_nums[i]] for i in range(0, len(files))]
	files = files

	# get rid of weird mac files
	for file in files:
		if file.endswith('DS_Store'): files.remove(file)

	# if no file number was provided to read then print files list
	if file_number == None: 
		print(files)
		return

	#f ind all dirs in scan folder
	file = files[file_number]
	for path, dirs, files in os.walk('../../Data/HDD/uCT/low_res/'+file+''):
		dirs = sorted(dirs)
		break

	# Find tif folder and if it doesnt exist read images in main folder
	tif = []
	for i in dirs: 
		if i.startswith('EK'):
			tif.append(i)
	if tif: tifpath = path+'/'+tif[0]+'/'
	else: tifpath = path+'/'

	print('tifpath:', tifpath)
	tifpath = Path(tifpath)
	files = sorted(tifpath.iterdir())
	images = [str(f) for f in files if f.suffix == '.tif']

	ct = []
	print('[CTFishPy] Reading uCT scan')
	if r:
		for i in tqdm(range(*r)):
			slice_ = cv2.imread(images[i])     
			# use provided scale metric to downsize image
			height  = int(slice_.shape[0] * scale / 100)
			width   = int(slice_.shape[1] * scale / 100)
			slice_ = cv2.resize(slice_, (width, height), interpolation = cv2.INTER_AREA)     
			ct.append(slice_)
		ct = np.array(ct)

	else:
		for index, i in enumerate(images):
			if 892 > index > 890 : continue
			slice_ = cv2.imread(i)     
			# use provided scale metric to downsize image
			height  = int(slice_.shape[0] * scale / 100)
			width   = int(slice_.shape[1] * scale / 100)
			slice_ = cv2.resize(slice_, (width, height), interpolation = cv2.INTER_AREA)     
			ct.append(slice_)
		ct = np.array(ct)

	# check if image is empty
	if np.count_nonzero(ct) == 0:
		raise ValueError('Image is empty.')

	# read xtekct
	path = Path(path) # change str path to pathlib format
	files = path.iterdir()

	return ct # ct: (slice, x, y, 3)
ct = read_dirty(0)
fish = 'multi_fish_projection'
x, y, z = ctreader.get_max_projections(ct)
cv2.imwrite(f'output/{fish}_x.png', x)
cv2.imwrite(f'output/{fish}_y.png', y)
cv2.imwrite(f'output/{fish}_z.png', z)


# for fish in range(40,500)
# 	ct, metadata = ctreader.read(fish)

# 	projection = ctreader.get_max_projections(ct)[0]

# 	angle = ctreader.spin(projection)
# 	print(projection.shape)

# 	print(angle)