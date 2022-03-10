"""
CTreader is the main class you use to interact with ctfishpy
"""

from .read_amira import read_amira
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import h5py
import codecs
from dotenv import load_dotenv
from copy import deepcopy
import json
import os
import napari


class CTreader:
	def __init__(self, data_path=None):
		# print(Path().resolve())
		with open('ctfishpy/Metadata/local_dataset_path.txt') as f:
			data_path = f.readlines()[0]
		print(data_path)

		if data_path == None:
			envpath = Path('ctfishpy/Metadata/local_dataset_path.txt')
			envpath.touch()

			print('[CTfishpy] local path file not found, please tell me the path to your dataset folder?')
			new_path = input('Path:')

			with open('ctfishpy/Metadata/local_dataset_path.txt', "w") as f:
				f.write(f"{new_path}")

			load_dotenv()
			data_path = os.environ.get('DATASET_PATH')

		if data_path:
			self.dataset_path = Path(data_path)
			low_res_clean_path = self.dataset_path / "low_res_clean/"
			nums = [int(path.stem) for path in low_res_clean_path.iterdir() if path.is_dir()]
			nums.sort()
			self.fish_nums = nums
		else:
			raise Exception('cant find data')

		self.master = pd.read_csv("ctfishpy/Metadata/uCT_mastersheet.csv")
		self.anglePath = Path("ctfishpy/Metadata/angles.json")
		self.centres_path = Path("ctfishpy/Metadata/centres_Otoliths.json")
		with open(self.centres_path, "r") as fp:
			self.manual_centers = json.load(fp)

	def mastersheet(self):
		return self.master

	def trim(self, m, col, values):
		"""
		Trim df to e.g. fish that are 12 years old
		Find all rows that have specified value in specified column
		e.g. find all rows that have 12 in column 'age'
		"""
		trimmed = m[m[col].isin(values)]

		return trimmed

	def list_numbers(self, m):
		# List numbers of fish in a dictionary after trimming
		return list(m.loc[:]["n"])

	def read_path(self, path, r=None):

		images = [str(i) for i in path.iterdir() if i.suffix == '.tiff']
		images.sort()

		

		ct = []
		print(f"[CTFishPy] Reading uCT scan. Fish: {path}")
		if r:
			for i in tqdm(range(*r)):
				tiffslice = tiff.imread(images[i])
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		else:
			for i in tqdm(images):
				tiffslice = tiff.imread(i)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		return ct

	def read(self, fish, r=None, align=False):
		"""
		Main function to read zebrafish from local dataset path specified in .env

		parameters
		fish : number of sample you want to read
		r : range of slices you want to read to save RAM
		align : manually aligns fish for dorsal fin to point upwards
		"""

		fishpath = self.dataset_path / "low_res_clean" / str(fish).zfill(3)
		tifpath = fishpath / "reconstructed_tifs"
		metadatapath = fishpath / "metadata.json"

		# Apologies this is broken but angles available in some metadata files (v4 dataset)
		# but not available on older dataset so can revert to using angle json
		if align:
			with open(self.anglePath, "r") as fp:
				angles = json.load(fp)
			angle = angles[str(fish)]

		stack_metadata = self.read_metadata(fish)
		# angle = stack_metadata['angle']

		# images = list(tifpath.iterdir())
		images = [str(i) for i in tifpath.iterdir()]
		images.sort()
		# print(images)

		ct = []
		print(f"[CTFishPy] Reading uCT scan. Fish: {fish}")
		if r:
			for i in tqdm(range(*r)):
				tiffslice = tiff.imread(images[i])
				if align == True:
					tiffslice = self.rotate_image(tiffslice, angle, is_label=False)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		else:
			for i in tqdm(images):
				tiffslice = tiff.imread(i)
				if align == True:
					tiffslice = self.rotate_image(tiffslice, angle, is_label=False)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		return ct, stack_metadata

	def read_range(self, n: int, center: list, roiSize: list):
		#only read range
		new_center = deepcopy(center)

		roiZ = roiSize[0]
		z_center = new_center[0]
		new_center[0] = int(roiSize[0]/2)
		ct, stack_metadata = self.read(n, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)
		ct = self.crop3d(ct, roiSize, center=new_center)
		return ct, stack_metadata

	def read_metadata(self, fish):
		"""
		Return metadata dictionary from each fish json
		"""
		fishpath = self.dataset_path / "low_res_clean" / str(fish).zfill(3)
		metadatapath = fishpath / "metadata.json"
		with metadatapath.open() as metadatafile:
			stack_metadata = json.load(metadatafile)
		return stack_metadata

	def read_label(self, organ, n, is_amira=True):
		"""
		Read and return hdf5 label files

		parameters
		organ : give string of organ you want to read, for now this is 'Otoliths' or 'Otoliths_unet2d'
		n : number of fish to get labels

		NOTE: This always reads labels aligned where dorsal fin is pointing upwards 
		so make you sure you align your scan when you read it

		"""

		# if organ not in ['Otoliths']:
		# 	raise Exception('organ not found')


		if n!=0 and is_amira==False:
			label_path = str(self.dataset_path / f'Labels/Organs/{organ}/{organ}.h5')
			print(f"[CTFishPy] Reading labels fish: {n} {label_path} ")

			with h5py.File(label_path, "r") as f:
				label = np.array(f[str(n)])
			

		elif n!=0 and is_amira==True:
			label_path = self.dataset_path / f'Labels/Organs/{organ}/{n}.am'
			print(f"[CTFishPy] Reading labels fish: {n} {label_path} ")
			
			label_dict = read_amira(label_path)
			label = label_dict['data'][-1]['data'].T

			# fix for different ordering from mariel labels
			mariel_samples	= [421,423,242,463,259,459,256,530,589] 
			if n in mariel_samples and organ == 'Otoliths':
				label[label==2]=1
				label[label==3]=2
				label[label==4]=3

		if organ == 'Otoliths':
			if is_amira:
				align = True if n in [78,200,218,240,277,330,337,341,462,464,364,385] else False # This is a fix for undergrad labelled data
			else:
				align = True
		elif organ == 'Otoliths_unet2d':
			align = False
		
		if align:
			# get manual alignment
			with open(self.anglePath, "r") as fp:
				angles = json.load(fp)
			angle = angles[str(n)]
			# stack_metadata = self.read_metadata(n)
			label = [self.rotate_image(i, angle, is_label=True) for i in label]
			label = np.array(label)

		print("Labels ready.")
		return label

	def write_label(self, organ, label, n, dtype='uint8'):
		'''
		Write label to organ hdf5

		parameters
		label : label to save as a numpy array
		n : number of fish, put n = 0 if label is a cc template
		'''
		folderPath =  Path(f'{self.dataset_path}/Labels/Organs/{organ}/')
		folderPath.mkdir(parents=True, exist_ok=True)
		path = Path(f'{self.dataset_path}/Labels/Organs/{organ}/{organ}.h5')
		if n == 0: path = Path(f'{self.dataset_path}/Labels/Templates/{organ}.h5')

		with h5py.File(path, "a") as f:
			# print(f.keys())
			# exit()
			dset = f.create_dataset(str(n), shape=label.shape, dtype=dtype, data = label, compression=1)

	def write_scan(self, dataset, scan, n, compression=1, dtype='uint16'):
		'''
		Write scan to hdf5

		parameters
		label = label to save as a numpy array
		put n =0 if label is a cc template
		'''
		folderPath = Path(f'{self.dataset_path}/Compressed/')
		folderPath.mkdir(parents=True, exist_ok=True)
		path = folderPath / f'{dataset}.h5'
		with h5py.File(path, 'a') as f:
			dset = f.create_dataset(name=str(n), data = scan, shape=scan.shape, dtype=dtype, compression=compression)
		
	def view(self, array: np.ndarray, label: np.ndarray=None):

		viewer = napari.view_image(array, name='Scan')

		if label is not None:
			viewer.add_image(label, opacity=0.5, name='label')

		napari.run()


	def read_max_projections(self, n):
		"""
		Return z,y,x which represent axial, saggital, and coronal max projections
		This reads them instead of generating them
		"""
		# import pdb; pdb.set_trace()
		dpath = str(self.dataset_path)
		z = cv2.imread(f"{dpath}/projections/z/z_{n}.png")
		y = cv2.imread(f"{dpath}/projections/y/y_{n}.png")
		x = cv2.imread(f"{dpath}/projections/x/x_{n}.png")
		return np.array([z, y, x])

	def make_max_projections(self, stack, label=None):
		"""
		Make z,y,x which represent axial, saggital, and coronal max projections
		"""
		# import pdb; pdb.set_trace()
		z = np.max(stack, axis=0)
		y = np.max(stack, axis=1)
		x = np.max(stack, axis=2)
		projections = np.array([z, y, x])

		if label:
			projections = self.label_projections(projections, label)

		return projections

	def label_projections(self, projections, label):
		label_projections_list = self.make_max_projections(label)

		for i, p in enumerate(projections):
			p[ label_projections_list[i] == 1 ]=[255,0,0]
			p[ label_projections_list[i] == 2 ]=[255,255,0]
			p[ label_projections_list[i] == 3 ]=[0,0,255]
			p[ label_projections_list[i] == 4 ]=[0,255,255]

		return projections

	def resize(self, img, percent=100):
		# use scipy zoom for this
		width = int(img.shape[1] * percent / 100)
		height = int(img.shape[0] * percent / 100)
		return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

	def to8bit(self, img):
		"""
		Change img from 16bit to 8bit by mapping the data range to 0 - 255
		"""
		if img.dtype == "uint16":
			new_img = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8)
			return new_img
		else:
			print("image already 8 bit!")
			return img

	def rotate_array(self, array, angle, is_label, center=None):
		new_array = []
		for a in array:
			a_rotated = self.rotate_image(a, angle=angle, is_label=is_label, center=center)
			new_array.append(a_rotated)
		return np.array(new_array)

	def rotate_image(self, image, angle, is_label, center=None):
		"""
		Rotate images properly using cv2.warpAffine
		since it provides more control eg over center

		parameters
		image : 2d np array
		angle : angle to spin
		center : provide center if you dont want to spin around true center
		"""
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		if center:
			image_center = center
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		# THIS HAS TO BE NEAREST NEIGHBOUR BECAUSE LABELS ARE CATEGORICAL
		if is_label:
			interpolation_flag = cv2.INTER_NEAREST
		else:
			interpolation_flag = cv2.INTER_LINEAR
		result = cv2.warpAffine(
			image, rot_mat, image.shape[1::-1], flags=interpolation_flag
		)
		return result

	def thresh_stack(self, stack, thresh_8):
		"""
		Threshold CT stack in 16 bits using numpy because it's faster
		provide threshold in 8bit since it's more intuitive, then convert to 16
		"""

		thresh_16 = thresh_8 * (65535 / 255)

		thresholded = []
		for slice_ in stack:
			new_slice = (slice_ > thresh_16) * slice_
			thresholded.append(new_slice)

		return np.array(thresholded, dtype='uint16')

	def thresh_img(self, img, thresh_8, is_16bit=False):
		"""
		Threshold CT img 
		Default is 8bit thresholding but make 16_bit=True if not
		"""
		#provide threshold in 8bit since it's more intuitive then convert to 16
		thresh_16 = thresh_8 * (65535 / 255)
		if is_16bit:
			thresh = thresh_16
		if not is_16bit:
			thresh = thresh_8
		new_img = (img > thresh) * img
		return new_img

	def saveJSON(self, nparray, jsonpath):
		"""
		A quick way to save nparrays as json
		"""
		json.dump(
			nparray,
			codecs.open(jsonpath, "w", encoding="utf-8"),
			separators=(",", ":"),
			sort_keys=True,
			indent=4,
		)  ### this saves the array in .json format

	def readJSON(self, jsonpath):
		"""
		Quickly read nparrays as json
		"""
		obj_text = codecs.open(jsonpath, "r", encoding="utf-8").read()
		obj = json.loads(obj_text)
		return np.array(obj)

	def crop3d(self, array, roiSize, center=None):
		roiZ, roiY, roiX = roiSize
		zl = int(roiZ / 2)
		yl = int(roiY / 2)
		xl = int(roiX / 2)

		if center == None:
			c = int(array.shape[0] / 2)
			center = [c, c, c]

		z, y, x = center
		z, y, x = int(z), int(y), int(x)
		array = array[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl]
		return array

	def crop_around_center3d(self, array, roiSize, roiZ=None, center=None):
		"""
		Crop around the center of 3d array
		You can specify the center of crop if you want
		Also possible to set different ROI size for XY and Z
		"""

		xl = int(roiSize[0] / 2)
		yl = int(roiSize[1] / 2)
		zl = int(roiZ / 2)

		if center == None:
			c = int(array.shape[0] / 2)
			center = [c, c, c]
		z, x, y = center
		array = array[z - zl : z + zl, x - xl : x + xl, y - yl : y + yl]
		return array

	def crop_around_center2d(self, array, center=None, roiSize=100):
		"""
		I have to crop a lot of images so this is a handy utility function
		If you don't provide a center this will crop around nominal center
		"""
		l = int(roiSize / 2)
		if center == None:
			t = int(array.shape[0] / 2)
			center = [t, t]
		x, y = center
		array = array[x - l : x + l, y - l : y + l]
		return array

	def getVol(self, label, metadata, nclasses):
		counts = np.array([np.count_nonzero(label == i) for i in range(1, nclasses+1)])
		voxel_size = float(metadata['VoxelSizeX']) * float(metadata['VoxelSizeY']) * float(metadata['VoxelSizeZ'])
		volumes = counts * voxel_size

		return volumes

	def getDens(self, scan, label, nclasses, dens_calib = 0.0000381475547417411):
		voxel_values = np.array([np.mean(scan[label == i]) for i in range(1, nclasses+1)])
		densities = voxel_values * dens_calib

		return densities
