from moviepy.editor import ImageSequenceClip
from .read_amira import read_amira
try: from ..viewer import * 
except: from ..viewer import cc_fixer, mainViewer, spinner
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import cv2
import h5py
import codecs
from dotenv import load_dotenv
import os


class CTreader:
	def __init__(self, data_path=None):
		load_dotenv()
		data_path = os.environ.get('DATASET_PATH')

		if data_path == None:
			envpath = Path('.env')
			envpath.touch()

			print('[CTfishpy] .env file not found, please tell me the path to your dataset folder?')
			new_path = input('Path:')

			with open(".env", "w") as f:
				f.write(f"DATASET_PATH={new_path}")

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
		self.centres_path = Path("ctfishpy/Metadata/cc_centres_Otoliths.json")
		with open(self.centres_path, "r") as fp:
			self.manual_centers = json.load(fp)

	def mastersheet(self):
		return self.master

	def trim(self, m, col, value):
		"""
		Trim df to e.g. fish that are 12 years old
		Find all rows that have specified value in specified column
		e.g. find all rows that have 12 in column 'age'
		"""
		# delete ones not in index
		index = list(m.loc[m[col]==value].index.values)
		trimmed = m.drop(set(m.index) - set(index))
		return trimmed

	def list_numbers(self, m):
		# List numbers of fish in a dictionary after trimming
		return list(m.loc[:]["n"])

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
					tiffslice = self.rotate_image(tiffslice, angle)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		else:
			for i in tqdm(images):
				tiffslice = tiff.imread(i)
				if align == True:
					tiffslice = self.rotate_image(tiffslice, angle)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

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

		NOTE: This always reads labels aligned that dorsal fin is pointing upwards 
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
		
		align = True if n in [78,200,218,240,277,330,337,341,462,464,364,385] else False # This is a fix for undergrad labelled data
		if align:
			# get manual alignment
			with open(self.anglePath, "r") as fp:
				angles = json.load(fp)
			angle = angles[str(n)]
			# stack_metadata = self.read_metadata(n)
			label = [self.rotate_image(i, angle) for i in label]
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

	def make_max_projections(self, stack):
		"""
		Make z,y,x which represent axial, saggital, and coronal max projections
		"""
		# import pdb; pdb.set_trace()
		z = np.max(stack, axis=0)
		y = np.max(stack, axis=1)
		x = np.max(stack, axis=2)
		return np.array([z, y, x])

	def view(self, ct, label=None, thresh=False):
		"""
		Main viewer using PyQt5
		"""
		mainviewer.mainViewer(ct, label, thresh)

	def spin(self, img, center=None, label=None, thresh=False):
		"""
		Manual spinner made to align fish
		"""
		angle = spinner(img, center, label, thresh)
		return angle

	def cc_fixer(self, fish):
		"""
		my localiser aka cc_fixer

		Positions that come from PyQt QPixmap are for some reason in y, x format
		]
		"""

		projections = self.read_max_projections(fish)
		[print(p.shape) for p in projections]
		positions = [cc_fixer.mainFixer(p) for p in projections]

		x = int(positions[0][1])
		y = int(positions[0][0])		
		z = int(positions[1][1])
		return [z, x, y]

	def resize(self, img, percent=100):
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

	def rotate_array(self, array, angle, center=None):
		new_array = []
		print('Rotating...')
		for a in array:
			a_rotated = self.rotate_image(a, angle=angle, center=center)
			new_array.append(a_rotated)
		return np.array(new_array)

	def rotate_image(self, image, angle, center=None):
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
		result = cv2.warpAffine(
			image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
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
		roiX, roiY, roiZ = roiSize
		xl = int(roiX / 2)
		yl = int(roiY / 2)
		zl = int(roiZ / 2)

		if center == None:
			c = int(array.shape[0] / 2)
			center = [c, c, c]

		z, x, y = center
		array = array[z - zl : z + zl, x - xl : x + xl, y - yl : y + yl]
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

	def make_gif(self, stack, file_name, fps = 10, label=None, scale=None):
		#decompose grayscale numpy array into RGB
		if stack.dtype == 'uint16': stack = ((stack - stack.min()) / (stack.ptp() / 255.0)).astype(np.uint8) 
		new_stack = np.array([np.stack((img,)*3, axis=-1) for img in stack], dtype='uint8')

		colors = [(0,0,0), (255,0,0), (255,255,0), (0,0,255)]

		if label is not None:
			for i in np.unique(label):
				if i != 0:
					new_stack[label==i] = colors[i]
		
		if scale is not None:
			im = new_stack[0]
			width = int(im.shape[1] * scale / 100)
			height = int(im.shape[0] * scale / 100)
			dim = (width, height)

			# resize image
			resized = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in new_stack]
			new_stack = resized
	

		# write_gif(new_stack, file_name, fps = fps)
		
		clip = ImageSequenceClip(list(new_stack), fps=fps)
		clip.write_gif(file_name, fps=fps)

	def getVol(self, label, metadata, nclasses):
		counts = np.array([np.count_nonzero(label == i) for i in range(1, nclasses+1)])
		voxel_size = float(metadata['VoxelSizeX']) * float(metadata['VoxelSizeY']) * float(metadata['VoxelSizeZ'])
		volumes = counts * voxel_size

		return volumes

	def getDens(self, scan, label, nclasses):
		dens_calib = 0.0000381475547417411
		voxel_values = np.array([np.mean(scan[label == i]) for i in range(1, nclasses+1)])
		densities = voxel_values * dens_calib

		return densities