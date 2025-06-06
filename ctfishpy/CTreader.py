"""
CTreader is the main class you use to interact with ctfishpy
"""

from sklearn.utils import deprecated
from .read_amira import read_amira
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import h5py
import json
import napari
import warnings
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import datetime
import time
from typing import Union

class CTreader:
	def __init__(self, data_path:Union[str, Path, None]=None):
		# print(Path().resolve())

		if data_path:
			self.dataset_path = Path(data_path)			
			
			self.dicoms_path = self.dataset_path / "DICOMS/"
			nums = [int(path.stem.split('_')[1]) for path in self.dicoms_path.iterdir()]
			nums.sort()
			self.fish_nums = nums
			self.master_path = self.dataset_path / "METADATA/uCT_mastersheet.csv"
			self.master = pd.read_csv(self.master_path, index_col='n')
			
			# TODO remake into ctreader.centers dict and select from bone first
			otolith_centers = self.master.otolith_center.to_dict()
			self.otolith_centers = {k: np.fromstring(c[1:-1], dtype='uint16', sep=' ') for k, c in otolith_centers.items() } # this is a fix to read a numpy array from a pandas df element
			jaw_centers = self.master.jaw_center.to_dict()
			self.jaw_centers = {k: np.fromstring(c[1:-1], dtype='uint16', sep=' ') for k, c in jaw_centers.items() } # this is a fix to read a numpy array from a pandas df element			

			# self.anglePath = Path(self.dataset_path / "METADATA/angles.json")
			# self.centres_path = Path(self.dataset_path / "METADATA/centres_Otoliths.json")
			# with open(self.centres_path, "r") as fp:
			# 	self.manual_centers = json.load(fp)

			self.dataset_initialised = True
		else:
			self.dataset_initialised = False
		
		self.bones = ["OTOLITHS", "JAW"]
		self.OTOLITHS = "OTOLITHS"
		self.JAW = "JAW"

	def metadata_tester():
		"""
		Test fish_nums in local dataset vs mastersheet
		test mastersheet NANs
		"""
		pass

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

	def read(self, fish:int):
		if self.dataset_initialised:
			start = time.time()
			scan = self.read_dicom(self.dicoms_path / f"ak_{fish}.dcm")
			end = time.time()
			# print(f"Reading dicom {fish} took {end-start} seconds") 
			return scan
		else:
			raise Exception("Dataset not initialised")

	def read_roi(self, fish:int, roi, center=None):
		if self.dataset_initialised:
			# start = time.time()
			scan = self.read_dicom(self.dicoms_path / f"ak_{fish}.dcm")

			scan = self.crop3d(scan, roi, center)

			# end = time.time()
			# print(f"Reading dicom {fish} took {end-start} seconds") 
			return scan
		else:
			raise Exception("Dataset not initialised")

	def read_metadata(self, fish:int, old_n = False) -> dict:
		return self.master.loc[fish].to_dict()

	def read_dicom(self, path, bits=16, dtype='uint16'):
		with pydicom.dcmread(path) as ds:
			ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
			data = ds.pixel_array
			data = data.astype(dtype)
				
		return data

	def write_dicom(self, path:str, array:np.ndarray):
		"""
		save monochrome dicom 
		this will auto determine 16 / 8 bit depth but will only accept np arrays in dtypes uint8 or uint16
		"""
		suffix = ".dcm"
		# Populate required values for file meta information
		file_meta = FileMetaDataset()
		file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
		file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
		file_meta.ImplementationClassUID = UID("1.2.3.4")
		file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

		# Create the FileDataset instance (initially no data elements, but file_meta
		# supplied)
		ds = FileDataset(path, {},
				 file_meta={}, preamble=b"\0" * 128)

		ds.PatientName = "123456"
		ds.PatientID = "123456"

		# Set creation date/time and endianness
		dt = datetime.datetime.now()
		ds.ContentDate = dt.strftime('%Y%m%d')
		timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
		ds.ContentTime = timeStr
		# Set the transfer syntax
		# Write as a different transfer syntax XXX shouldn't need this but pydicom
		# 0.9.5 bug not recognizing transfer syntax
		# ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
		ds.is_little_endian = True
		ds.is_implicit_VR = True # make reader lookup from dict
		ds.SamplesPerPixel = 1
		ds.PixelRepresentation = 1
		ds.PhotometricInterpretation = 'MONOCHROME1'
		ds.NumberOfFrames = array.shape[0]
		ds.Rows = array.shape[1]
		ds.Columns = array.shape[2]
		if array.dtype == "uint16": bits = 16
		if array.dtype == "uint8": bits = 8
		ds.BitsAllocated = bits
		ds.BitsStored = bits
		# missing: BitsAllocated, Rows, Columns, SamplesPerPixel, PhotometricInterpretation, PixelRepresentation, BitsStored

		ds.PixelData = array.tobytes()

		ds.save_as(path, write_like_original=True)

	def read_tif(self, path, r=None):

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

	@deprecated
	def old_read(self, fish, r=None, align=False):
		"""
		Main function to read zebrafish from local dataset path specified during initialisation

		parameters
		fish : index of sample you want to read
		r : range of slices you want to read to save RAM
		align : manually aligns fish for dorsal fin to point upwards
		"""

		fishpath = self.low_res_clean_path / str(fish).zfill(3)
		tifpath = fishpath / "reconstructed_tifs"
		metadatapath = fishpath / "metadata.json"

		# Apologies this is broken but angles available in some metadata files (v4 dataset)
		# but not available on older dataset so can revert to using angle json
		if align:
			with open(self.anglePath, "r") as fp:
				angles = json.load(fp)
			angle = angles[str(fish)]['angle']
			center = angles[str(fish)]['center']
		else:
			angle=0
			center=None

		print(f"ANGLE: {angle}{center}")

		stack_metadata = self.read_metadata(fish)
		# angle = stack_metadata['angle']
		# center = stack_metadata['center']

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
					tiffslice = self.rotate_image(tiffslice, angle, is_label=False, center=center)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		else:
			for i in tqdm(images):
				tiffslice = tiff.imread(i)
				if align == True:
					tiffslice = self.rotate_image(tiffslice, angle, is_label=False, center=center)
				ct.append(tiffslice)
			ct = np.array(ct, dtype='uint16')

		return ct, stack_metadata

	def read_hdf5(self, path, index):
		with h5py.File(path, "r") as f:
			array = np.array(f[str(index)])
		return array

	def read_label(self, bone, n, is_amira=False, is_tif=False, name=None):
		"""
		Read and return hdf5 label files

		parameters
		bone : give string of bone you want to read, for now this is 'Otoliths'
		n : number of fish to get label

		"""

		if is_amira==False and is_tif == False:
			if name is None: name = bone
			label_path = str(self.dataset_path / f'LABELS/{bone}/{name}.h5')

			with h5py.File(label_path, "r") as f:
				label = np.array(f[str(n)])

		elif is_amira==True:
			label_path = str(self.dataset_path / f'LABELS/{bone}/{n}.am')
			label_dict = read_amira(label_path)
			label = label_dict['data'][-1]['data'].T
			
		elif is_tif==True:
			label_path = str(self.dataset_path / f'LABELS/{bone}/{n}.tif')
			label = tiff.imread(label_path)

		return label

	def get_hdf5_keys(self, path) -> list:

		with h5py.File(path, "r") as f:
			keys = list(f.keys())
			nums = keys
		nums = [int(n) for n in nums]
		nums.sort()
		return nums

	# @deprecated
	def old_read_label(self, bone, n, is_amira=True):
		"""
		Read and return hdf5 label files

		parameters
		bone : give string of bone you want to read, for now this is 'Otoliths' or 'Otoliths_unet2d'
		n : number of fish to get labels

		TODO clean marielle and zack otolith labels into new hdf5 or can i write amira?

		NOTE: This always reads labels aligned where dorsal fin is pointing upwards 
		so make you sure you align your scan when you read it

		"""

		# if bone not in ['Otoliths']:
		# 	raise Exception('bone not found')


		if is_amira==False:
			label_path = str(self.dataset_path / f'LABELS/{bone}/{bone}.h5')

			with h5py.File(label_path, "r") as f:
				label = np.array(f[str(n)])
			

		elif is_amira==True:
			label_path = self.dataset_path / f'LABELS/{bone}/{n}.am'
			
			label_dict = read_amira(label_path)
			label = label_dict['data'][-1]['data'].T

			# fix for different ordering from mariel labels
							 # [421,423,242,463,259,459,461]
			mariel_samples	= [421,423,242,463,259,459,256,530,589] 
			if n in mariel_samples and bone == 'Otoliths':
				label[label==2]=1
				label[label==3]=2
				label[label==4]=3

		if bone == 'Otoliths':
			if is_amira:
				align = True if n in [78,200,218,240,277,330,337,341,462,464,364,385] else False # This is a fix for undergrad labelled data
			else:
				align = True
		elif bone == 'Otoliths_unet2d':
			align = False
		
		if align:
			# get manual alignment
			with open(self.anglePath, "r") as fp:
				angles = json.load(fp)
			angle = angles[str(n)]['angle']
			center = angles[str(n)]['center']
			# stack_metadata = self.read_metadata(n)
			label = self.rotate_array(label, angle, is_label=True, center=center)
			label = np.array(label)

		return label

	def write_label(self, bone, label, n, name=None, rewrite=False, dtype='uint8'):
		''' 
		Write label to bone hdf5

		parameters
		label : label to save as a numpy array
		n : number of fish, put n = 0 if label is a cc template
		'''

		if name is None:
			name = bone

		folderPath =  Path(f'{self.dataset_path}/LABELS/{bone}/')
		folderPath.mkdir(parents=True, exist_ok=True)
		path = Path(f'{self.dataset_path}/LABELS/{bone}/{name}.h5')

		with h5py.File(path, "a") as f:
			# print(f.keys())
			if rewrite:
				dset[str(n)] = label
			else:
				dset = f.create_dataset(str(n), shape=label.shape, dtype=dtype, data = label, compression=1)

	def write_scan(self, dataset, scan, n, compression=1, dtype='uint16'):
		'''
		Write scan to hdf5

		parameters
		label = label to save as a numpy array
		put n =0 if label is a cc template
		compression 9 or 1 smaller?
		'''
		folderPath = Path(f'{self.dataset_path}/Compressed/')
		folderPath.mkdir(parents=True, exist_ok=True)
		path = folderPath / f'{dataset}.h5'
		with h5py.File(path, 'a') as f:
			dset = f.create_dataset(name=str(n), data = scan, shape=scan.shape, dtype=dtype, compression=compression)
		
	def view(self, array: np.ndarray, label: np.ndarray=None):

		viewer = napari.view_image(array, name='Scan')

		if label is not None:
			# viewer.add_image(label, opacity=0.5, name='label')
			viewer.add_labels(label, opacity=0.5, name="Label")
		napari.run()

	def read_max_projections(self, n):
		"""
		Return z,y,x which represent axial, saggital, and coronal max projections
		This reads them instead of generating them
		"""
		# import pdb; pdb.set_trace()
		dpath = str(self.dataset_path)
		z = cv2.imread(f"{dpath}/PROJECTIONS/Z/z_{n}.png")
		y = cv2.imread(f"{dpath}/PROJECTIONS/Y/y_{n}.png")
		x = cv2.imread(f"{dpath}/PROJECTIONS/X/x_{n}.png")
		return np.array([z, y, x], dtype=object)

	def make_max_projections(self, stack):
		"""
		Make z,y,x which represent axial, saggital, and coronal max projections

		if label provided it will color the scan for figures
		"""
		# import pdb; pdb.set_trace()

		z = np.max(stack, axis=0)
		y = np.max(stack, axis=1)
		x = np.max(stack, axis=2)
		projections = [z,y,x] #np.array([z, y, x])

		return projections

	def label_projections(self, scan_proj, mask_proj):
		scan_proj = [np.array(cv2.cvtColor(s, cv2.COLOR_GRAY2RGB), dtype=np.uint8) for s in scan_proj]		

		for i, p in enumerate(scan_proj):
			p[mask_proj[i] == 1 ]=[255,0,0]
			p[mask_proj[i] == 2 ]=[255,255,0]
			p[mask_proj[i] == 3 ]=[0,0,255]
			p[mask_proj[i] == 4 ]=[0,255,255]

		return [np.array(s, dtype=np.uint8) for s in scan_proj]

	def resize(self, img, scale=100):
		# use scipy ndimage zoom
		width = int(img.shape[1] * scale / 100)
		height = int(img.shape[0] * scale / 100)
		return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

	def to8bit(self, array:np.ndarray):
		"""
		Change array from 16bit to 8bit by mapping the data range to 0 - 255

		*NOTE* This does not convert to 8bit normally and is for GUI functions
		"""
		if array.dtype == "uint16":
			new_array = ((array - array.min()) / (array.ptp() / 255.0)).astype(np.uint8)
			return new_array
		else:
			raise Exception("image already 8 bit!")
			return new_array

	def rotate_array(self, array, angle, is_label=False, center=None):
		"""
		Rotate using affine transformation
		"""
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
			image_center = tuple(center)
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

	def crop3d(self, array, roiSize, center=None):
		roiZ, roiY, roiX = roiSize
		zl = int(roiZ / 2)
		yl = int(roiY / 2)
		xl = int(roiX / 2)

		if center is None:
			center = [int(array.shape[0]/2), int(array.shape[1]/2), int(array.shape[1]/2)]

		z, y, x = center
		z, y, x = int(z), int(y), int(x)

		ctz, cty, ctx = array.shape

		rectz = [z-zl, z+zl]
		recty = [y-yl, y+yl]
		rectx = [x-xl, x+xl]

		shifted = False
		# if statements to shift crop inside ct window
		if rectz[0] < 0:
			shiftz = -rectz[0]
			rectz[0] = 0
			rectz[1] = rectz[1] + shiftz
			shifted = True

		if rectz[1] > ctz:
			shiftz = rectz[1] - ctz
			rectz[1] = ctz
			rectz[0] = rectz[0] - shiftz
			shifted = True

		if recty[0] < 0:
			shifty = -recty[0]
			recty[0] = 0
			recty[1] = recty[1] + shifty
			shifted = True

		if recty[1] > cty:
			shifty = recty[1] - cty
			recty[1] = cty
			recty[0] = recty[0] - shifty
			shifted = True

		if rectx[0] < 0:
			shiftx = -rectx[0]
			rectx[0] = 0
			rectx[1] = rectx[1] + shiftx
			shifted = True

		if rectx[1] > ctx:
			shiftx = rectx[1] - ctx
			rectx[1] = ctx
			rectx[0] = rectx[0] - shiftx
			shifted = True
		if shifted:
			message = f"Warning, the requested crop indices are outside the target area so I have shifted them for you \n {rectz, rectx, recty}"
			warnings.warn(message)
			
		#TODO print new  rect  x  etc  and warn when  triggered
		# new_array = array[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl]+0 # add 0 to create new copy and be able to delete old array if need be
		new_array = array[rectz[0] : rectz[1], recty[0] : recty[1], rectx[0] : rectx[1],]+0 # add 0 to create new copy and be able to delete old array if need be
		return new_array

	def uncrop3d(self, old_array, roi, center=None):
		array = np.zeros_like(old_array)
		roiZ, roiY, roiX = roi.shape
		zl = int(roiZ / 2)
		yl = int(roiY / 2)
		xl = int(roiX / 2)

		if center is None:
			center = [int(array.shape[0]/2), int(array.shape[1]/2), int(array.shape[1]/2)]

		z, y, x = center
		z, y, x = int(z), int(y), int(x)

		ctz, cty, ctx = array.shape

		rectz = [z-zl, z+zl]
		recty = [y-yl, y+yl]
		rectx = [x-xl, x+xl]

		shifted = False
		# if statements to shift crop inside ct window
		if rectz[0] < 0:
			shiftz = -rectz[0]
			rectz[0] = 0
			rectz[1] = rectz[1] + shiftz
			shifted = True

		if rectz[1] > ctz:
			shiftz = rectz[1] - ctz
			rectz[1] = ctz
			rectz[0] = rectz[0] - shiftz
			shifted = True

		if recty[0] < 0:
			shifty = -recty[0]
			recty[0] = 0
			recty[1] = recty[1] + shifty
			shifted = True

		if recty[1] > cty:
			shifty = recty[1] - cty
			recty[1] = cty
			recty[0] = recty[0] - shifty
			shifted = True

		if rectx[0] < 0:
			shiftx = -rectx[0]
			rectx[0] = 0
			rectx[1] = rectx[1] + shiftx
			shifted = True

		if rectx[1] > ctx:
			shiftx = rectx[1] - ctx
			rectx[1] = ctx
			rectx[0] = rectx[0] - shiftx
			shifted = True
		if shifted:
			message = f"Warning, the requested crop indices are outside the target area so I have shifted them for you \n {rectz, rectx, recty}"
			warnings.warn(message)
			
		#TODO print new  rect  x  etc  and warn when  triggered
		# new_array = array[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl]+0 # add 0 to create new copy and be able to delete old array if need be
		array[rectz[0] : rectz[1], recty[0] : recty[1], rectx[0] : rectx[1],] = roi # add 0 to create new copy and be able to delete old array if need be
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
		"""
		This will not measure background
		"""
		counts = np.array([np.count_nonzero(label == i) for i in range(1, nclasses+1)])
		voxel_size = float(metadata['VoxelSizeX']) * float(metadata['VoxelSizeY']) * float(metadata['VoxelSizeZ'])
		volumes = counts * voxel_size

		return volumes

	def getDens(self, scan, label, nclasses, dens_calib = 0.0000381475547417411):
		"""This will not measure background

		Args:
			scan (_type_): _description_
			label (_type_): _description_
			nclasses (_type_): _description_
			dens_calib (float, optional): _description_. Defaults to 0.0000381475547417411.

		Returns:
			_type_: _description_
		"""
		voxel_values = np.array([np.mean(scan[label == i]) for i in range(1, nclasses+1)])
		densities = voxel_values * dens_calib

		return densities

	def localise(self, projections:list=None, scan:np.ndarray=None, to_use:list=[0,1,2]):
		"""
		to_use is which projections to use, z, x, y
		"""
		from .GUI import create_localiser

		if len(to_use) < 2: raise ValueError("must use at least 2 projections")

		pos1, pos2, pos3 = [0,0], [0,0], [0,0]
		if isinstance(projections, (list, np.ndarray)):
			# projections = np.array([cv2.cvtColor(p, cv2.COLOR_GRAY2RGB) for p in projections])

			if 0 in to_use:
				viewer = napari.Viewer()
				m = {'pos': 0, 'og':projections[0]}
				layer = viewer.add_image(projections[0], metadata=m, name='scan')
				create_localiser(viewer, layer)
				viewer.show(block=True)
				metadata = layer.metadata
				pos1 = metadata['pos']
			
			if 1 in to_use:
				viewer = napari.Viewer()
				m = {'pos': 0, 'og':projections[1]}
				layer = viewer.add_image(projections[1], metadata=m, name='scan')
				create_localiser(viewer, layer)
				viewer.show(block=True)
				metadata = layer.metadata
				pos2 = metadata['pos']

			if 2 in to_use:
				viewer = napari.Viewer()
				m = {'pos': 0, 'og':projections[2]}
				layer = viewer.add_image(projections[2], metadata=m, name='scan')
				create_localiser(viewer, layer)
				viewer.show(block=True)
				metadata = layer.metadata
				pos3 = metadata['pos']

			final_center = [self.int_mean(pos2[1], pos3[1]), self.int_mean(pos1[1], pos3[0]), self.int_mean(pos1[0], pos2[0]),]

			return final_center

		elif isinstance(scan, np.ndarray):
			scan = self.to8bit(scan)
			scan = np.array([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in scan])
			projection = np.max(scan, axis=0)
			raise ValueError("localise from array not ready")

		else:
			raise ValueError("please provide either list of projections or np array of scan")

	def int_mean(self, a,b,):
		if a == 0 and b == 0: raise ValueError("both 0")
		elif a == 0: a = b
		elif b == 0: b = a
		
		return int((a+b)/2)


	def label_array(self, scan, label=None):

		from .GUI import create_labeller

		if label is None:
			label = np.zeros_like(scan)

		assert(scan.shape == label.shape)

		viewer = napari.Viewer()
		layer = viewer.add_image(scan)
		layer.metadata = {"point": None, "history": np.stack([label]), "slice": 0}
		label_layer = viewer.add_labels(label)

		create_labeller(viewer, layer, label_layer)

		viewer.show(block=True)

		return label_layer.data

	def insert_a_in_b(self, a: np.ndarray, b: np.ndarray, center=None):

		roiZ, roiY, roiX = a.shape
		zl, yl, xl = int(roiZ / 2), int(roiY / 2), int(roiX / 2)
 
		if center is None:
			b_center = [int(b.shape[2] / 2), int(b.shape[3] / 2), int(b.shape[4] / 2)]
			z, y, x = b_center
		else:
			z, y, x = center
		z, y, x = int(z), int(y), int(x)

		print(z, y, x, center, zl, yl, xl)
		print(a.shape, b.shape)
		print(z - zl, z + zl, y - yl, y + yl, x - xl, x + xl)

		b[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl] = a

		return b

	def plot_side_by_side(self, array, label):
		array_projections = self.make_max_projections(array)
		label_projections = self.make_max_projections(label)
		labelled_projections = self.label_projections(array_projections, label_projections)
		sidebyside = np.concatenate(labelled_projections[0:2], 0)

		return sidebyside