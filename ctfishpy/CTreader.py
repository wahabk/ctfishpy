from dotenv import load_dotenv
from .GUI import mainviewer
from .GUI import spinner
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import cv2
import h5py
import codecs
import os


class CTreader():
	def __init__(self):
		# Use a local .env file to set where dataset is on current machine
		# This .env file is not uploaded by git
		load_dotenv()
		self.dataset_path = Path(os.getenv('DATASET_PATH'))
		self.master = pd.read_csv('./uCT_mastersheet.csv')
		nums = [int(path.stem) for path in self.dataset_path.iterdir() if path.is_dir()]
		nums.sort()
		self.fish_nums = nums

	def mastersheet(self):
		return self.master

	def trim(self, m, col, value):
		# Trim df to e.g. fish that are 12 years old
		# Find all rows that have specified value in specified column
		# e.g. find all rows that have 12 in column 'age'
		index = list(m.loc[m[col]==value].index.values)
		# delete ones not in index
		trimmed = m.drop(set(m.index) - set(index))
		return trimmed

	def list_numbers(self, m):
		# List numbers of fish in a dictionary after trimming
		return list(m.loc[:]['n'])

	def read(self, fish, r=None, align=False):
		'''
		Main function to read zebrafish from local dataset path specified in .env
		'''

		fishpath = self.dataset_path / str(fish).zfill(3) 
		tifpath = fishpath / 'reconstructed_tifs'
		metadatapath = fishpath / 'metadata.json'
		with open('ctfishpy/angles.json', 'r') as fp:
			angles = json.load(fp)
		angle = angles[str(fish)]

		stack_metadata = self.read_metadata(fish)

		#images = list(tifpath.iterdir())
		images = [str(i) for i in tifpath.iterdir()]
		images.sort()

		ct = []
		print(f'[CTFishPy] Reading uCT scan. Fish: {fish}')
		if r:
			for i in tqdm(range(*r)):
				tiffslice = tiff.imread(images[i])
				if align == True: 
					tiffslice = self.rotate_image(tiffslice, angle)
				ct.append(tiffslice)
			ct = np.array(ct)

		else:
			for i in tqdm(images):
				tiffslice = tiff.imread(i)
				if align == True: 
					tiffslice = self.rotate_image(tiffslice, angle)
				ct.append(tiffslice)
			ct = np.array(ct)

		return ct, stack_metadata

	def read_metadata(self, fish):
		'''
		Return metadata dictionary from each fish json
		'''
		fishpath = self.dataset_path / str(fish).zfill(3) 
		metadatapath = fishpath / 'metadata.json'
		with metadatapath.open() as metadatafile:
			stack_metadata = json.load(metadatafile)
		return stack_metadata

	def view(self, ct, label = None, thresh = False):
		'''
		Main viewer using PyQt5
		'''
		mainviewer.mainViewer(ct, label, thresh)

	def spin(self, img, center, label = None, thresh = False):
		'''
		Manual spinner made to align fish
		'''
		angle = spinner(img, center, label, thresh)
		return angle

	def read_label(self, labelPath, align=False, n=None, manual=True):
		'''
		Read and return hdf5 label files 

		TODO Automatically find labelPaths stored in .Data/Labels/organ/fishnums

		'''
		
		print('[CTFishPy] Reading labels...')

		# Use h5py module to read labelpath and extract pure numpy array
		f = h5py.File(labelPath, 'r') 
		
		if manual: label = np.array(f['t0']['channel0'])
		else: 
			label = np.array(f['t0'])
		f.close()
		if align: 
			stack_metadata = self.read_metadata(n)
			label = [self.rotate_image(i, stack_metadata['angle']) for i in label]
			label = np.array(label)
		print('Labels ready.')
		return label

	def to8bit(self, img):
		'''
		Change img from 16bit to 8bit
		'''
		if img.dtype == 'uint16':
			new_img = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8) 
			return new_img
		else:
			print('image already 8 bit!')
			return img

	def rotate_image(self, image, angle, center=None):
		'''
		Rotate images properly using cv2.warpAffine 
		since it provides more control eg over center
		'''
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		if center: image_center = center
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
		return result

	def thresh_stack(self, stack, thresh_8):
		'''
		Threshold CT stack in 16 bits using numpy because it's faster
		provide threshold in 8bit since it's more intuitive then convert to 16
		'''

		thresh_16 = thresh_8 * (65535/255)

		thresholded = []
		for slice_ in stack:
			new_slice = (slice_ > thresh_16) * slice_
			thresholded.append(new_slice)
			
		return np.array(thresholded)

	def thresh_img(self, img, thresh_8, is_16bit=False):
		'''
		Threshold CT img in 16 bits using numpy because it's faster
		provide threshold in 8bit since it's more intuitive then convert to 16
		'''

		thresh_16 = thresh_8 * (65535/255)
		if is_16bit: thresh = thresh_16
		if is_16bit: thresh = thresh_8
		new_img = (img > thresh) * img
		return new_img

	def saveJSON(self, nparray, jsonpath):
		'''
		Just a quick way to save nparrays as json
		'''
		json.dump(nparray, codecs.open(jsonpath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

	def readJSON(self, jsonpath):
		'''
		Just a quick way to read nparrays as json
		'''
		obj_text = codecs.open(jsonpath, 'r', encoding='utf-8').read()
		obj = json.loads(obj_text)
		return np.array(obj)

	def get_max_projections(self, n):
		'''
		Return x, y, z which represent axial, saggital, and coronal max projections
		This reads them instead of generating them
		'''
		#import pdb; pdb.set_trace()
		x = cv2.imread(f'Data/projections/x/{n}.png')
		y = cv2.imread(f'Data/projections/y/{n}.png')
		z = cv2.imread(f'Data/projections/z/{n}.png')
		return [x, y, z]

	def make_max_projections(self, stack):
		'''
		Make x, y, z which represent axial, saggital, and coronal max projections
		'''
		#import pdb; pdb.set_trace()
		x = np.max(stack, axis=0)
		y = np.max(stack, axis=1)
		z = np.max(stack, axis=2)
		return [x, y, z]

	def resize(self, img, percent=100):
		width = int(img.shape[1] * percent / 100)
		height = int(img.shape[0] * percent / 100)
		return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

	def crop_around_center3d(self, array, center=None, roiSize=50, roiZ=None):
		'''
		Crop around the center of 3d array
		You can specify the center of crop if you want
		Also possible to set different ROI size for XY and Z
		'''
		
		l = int(roiSize/2)
		if roiZ: zl = int(roiZ/2)
		else: zl = l
		if center == None:
			c = int(array.shape[0]/2)
			center = [c, c, c]
		z, x, y  = center
		array = array[z-zl:z+zl, x-l:x+l, y-l:y+l]
		return array

	def crop_around_center2d(self, array, center=None, roiSize=100):
		'''
		I have to crop a lot of images so this is a handy utility function
		If you don't provide a center of rotation this will rotate around nominal center
		'''
		l = int(roiSize/2)
		if center == None:
			t = int(array.shape[0]/2)
			center = [t, t]
		x, y  = center
		array = array[x-l:x+l, y-l:y+l]
		return array
