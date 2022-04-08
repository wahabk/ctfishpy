from cv2 import circle
from qtpy.QtCore import QSettings
from pathlib2 import Path
from tqdm import tqdm
import tifffile as tiff
import pandas as pd
import numpy as np 
import json
import cv2
import h5py
import gc
import napari
from .GUI import tubeDetector, create_orderLabeller, create_spinner

class Lumpfish():
	
	def __init__(self):
		# Use a local .env file to set where dataset is on current machine
		# This .env file is not uploaded by git
		# load_dotenv()
		# self.dataset_path = Path(os.getenv("DATASET_PATH"))
		# self.master = pd.read_csv("ctfishpy/Metadata/uCT_mastersheet.csv")
		# low_res_clean_path = self.dataset_path / "low_res_clean/"
		# nums = [int(path.stem) for path in low_res_clean_path.iterdir() if path.is_dir()]
		# nums.sort()
		# self.fish_nums = nums
		self.anglePath = "ctfishpy/Metadata/angles.json"
		self.fishnums = np.arange(40,639)

	def mastersheet(self):
		#to count use master['age'].value_counts()
		return pd.read_csv('./uCT_mastersheet.csv')

	def read_tiff(self, path, r = None, scale = 40, get_metadata=False):
		#TODO merge with read dirty

		tifpath = Path(path)
		files = sorted(tifpath.iterdir())
		images = [str(f) for f in files if f.suffix == '.tif']
		# images.pop(891)

		ct = []
		print(f'[CTFishPy] Reading uCT scan: {path}')
		if r:
			for i in tqdm(range(*r)):
				# print(images[i])
				tiffslice = tiff.imread(images[i])
				ct.append(tiffslice)
			ct = np.array(ct)

		else:
			for i in tqdm(images):
				tiffslice = tiff.imread(i)
				ct.append(tiffslice)
			ct = np.array(ct)
		print(ct.shape)

		if scale:
			ct = self.rescale(ct, scale)

		if get_metadata:
			# TODO go path above, then I can merge this with read dirty
			# read xtekct
			path = Path(path) # change str path to pathlib format
			files = path.iterdir()
			xtekctpath = [str(f) for f in files if f.suffix == '.xtekct'][0]

			# check if xtekct exists
			if not Path(xtekctpath).is_file():
				raise Exception("[CTFishPy] XtekCT file not found. ")
			
			xtekct = QSettings(xtekctpath, QSettings.IniFormat)
			x_voxelsize = xtekct.value('XTekCT/VoxelSizeX')
			y_voxelsize = xtekct.value('XTekCT/VoxelSizeY')
			z_voxelsize = xtekct.value('XTekCT/VoxelSizeZ')

			metadata = {'path': str(path), 
						'scale' : scale,
						'x_voxel_size' : x_voxelsize,
						'y_voxel_size' : y_voxelsize,
						'z_voxel_size' : z_voxelsize}

			return ct, metadata # ct: (slice, x, y, 3)

		else:
			return ct

	def rescale(self, scan:np.ndarray, scale:int):
		if scale == 100: return scan

		new_ct = []
		for slice_ in scan:
			height  = int(slice_.shape[0] * scale / 100)
			width   = int(slice_.shape[1] * scale / 100)
			slice_ = cv2.resize(slice_, (width, height), interpolation = cv2.INTER_AREA)     
			new_ct.append(slice_)
		ct = np.array(new_ct)

		return ct

	def read_dirty(self, path, r = None, scale = 100):

		path = Path(path)
		dirs = [x for x in path.iterdir() if x.is_dir()]
		dirs = sorted(dirs)
		# print(dirs)
		
		# Find tif folder and if it doesnt exist read images in main folder
		tif = []
		for i in dirs: 
			if 'tifs' in str(i):
				tif.append(i)
		if tif: tifpath = path / tif[0]
		else: tifpath = path

		print('tifpath:', tifpath)
		tifpath = Path(tifpath)
		files = sorted(tifpath.iterdir())
		images = [str(f) for f in files if f.suffix == '.tif']

		ct = []
		print('[CTFishPy] Reading uCT scan')
		if r:
			for i in tqdm(range(*r)):
				slice_ = tiff.imread(images[i])     
				ct.append(slice_)
			ct = np.array(ct)

		else:
			for i in tqdm(images):
				slice_ = tiff.imread(i)       
				ct.append(slice_)
			ct = np.array(ct)

		if scale != 100:
			ct = self.rescale(ct)

		# read xtekct
		path = Path(path) # change str path to pathlib format
		files = path.iterdir()
		xtekctpath = [str(f) for f in files if f.suffix == '.xtekct'][0]

		# check if xtekct exists
		if not Path(xtekctpath).is_file():
			raise Exception("[CTFishPy] XtekCT file not found. ")
		
		xtekct = QSettings(xtekctpath, QSettings.IniFormat)
		x_voxelsize = xtekct.value('XTekCT/VoxelSizeX')
		y_voxelsize = xtekct.value('XTekCT/VoxelSizeY')
		z_voxelsize = xtekct.value('XTekCT/VoxelSizeZ')

		metadata = {'path': str(path), 
					'scale' : scale,
					'x_voxel_size' : x_voxelsize,
					'y_voxel_size' : y_voxelsize,
					'z_voxel_size' : z_voxelsize}

		return ct , metadata # ct: (slice, x, y, 3)
			
	def crop(self, ct, circles, scale = [40, 40]):
		'''
		this is so ugly :( im sorry             
		
		crop ct stack to circles provided in order
		
		find scale at which tubes detected and scale of current image
		
		remember that pyqt qpixmap returns locations in y,x instead of x,y
		scale = [from,to]

		
		'''
		
		scale_factor = scale[1]/scale[0]
		circles = [[int(x*scale_factor), int(y*scale_factor), int(r*scale_factor)] for x, y, r in circles]
		cropped_CTs = []
		ctx = ct.shape[2]
		cty = ct.shape[1]

		for x, y, r in circles:
			cropped_stack = []

			rectx = [x - r, x + r]
			recty = [y - r, y + r]
			
			# if statements to shift crop inside ct window
			if rectx[0] < 0:
				shiftx = -rectx[0]
				rectx[0] = 0
				rectx[1] = rectx[1] + shiftx
				# print(shiftx, rectx)

			if rectx[1] > ctx:
				shiftx = rectx[1] - ctx
				rectx[1] = ctx
				rectx[0] = rectx[0] - shiftx
				# print(shiftx, rectx)

			if recty[0] < 0:
				shifty = -recty[0]
				recty[0] = 0
				recty[1] = recty[1] + shifty
				# print(shifty, recty)

			if recty[1] > cty:
				shifty = recty[1] - cty
				recty[1] = cty
				recty[0] = recty[0] - shifty
				# print(shifty, recty)
			
			cropped_CTs.append(ct[:, recty[0] : recty[1], rectx[0] : rectx[1]])
			gc.collect()
			del rectx
			del recty
		return cropped_CTs

	def saveCrop(self, n, ordered_circles, metadata):
		fishnums = np.arange(40,639)
		number = fishnums[n]
		order = self.fish_order_nums[n]
		crop_data = {
			'n'                 : f'{order[0]}-{order[len(order)-1]}',
			'ordered_circles'   : ordered_circles.tolist(),
			'scale'             : metadata['scale'],
			'path'              : metadata['path']
		}
		jsonpath = metadata['path']+'/crop_data.json'
		with open(jsonpath, 'w') as o:
			json.dump(crop_data, o)
		backuppath = f'./output/Crops/{order[0]}-{order[len(order)-1]}_crop_data.json'
		with open(backuppath, 'w') as o:
			json.dump(crop_data, o)

	def readCrop(self, number):
		files = pd.read_csv('../../Data/HDD/uCT/filenames_low_res.csv', header = None)
		files = files.values.tolist()
		crop_path = '../../Data/HDD/uCT/low_res/'+files[number][0]+'/crop_data.json'
		with open(crop_path) as f:
			crop_data = json.load(f)
		return crop_data

	def write_metadata(self, n, input):
		'''
		metadata = {
			'N'    : None,
			'Skip'     : None,
			'Age'      : None,
			'Genotype'   : None,
			'Strain'     : None,
			'Name'     : None,
			'VoxelSizeX' : None,
			'VoxelSizeY' : None,
			'VoxelSizeZ' : None
		}
		'''
		#n = self.fishnums[n]
		fishpath = Path(f'../../Data/HDD/uCT/low_res_clean/{str(n).zfill(3)}/')
		jsonpath = fishpath / 'metadata.json'
		jsonpath.touch()

		'''
		# old stuff to dynamically add metadata to existing files
		with open(jsonpath) as f:
			metadata = json.load(f)

		for key in list(input.keys()):
			metadata[key] = input[key]
		'''

		# just dump input for now
		with open(jsonpath, 'w') as o:
			json.dump(input, o)

	def append_metadata(self, n, inputDict):
		metadataPath = f'../../Data/HDD/uCT/low_res_clean/{str(n).zfill(3)}/metadata.json'
		with open(metadataPath) as f:
			data = json.load(f)
		data.update(inputDict)
		with open(metadataPath, 'w') as f:
			json.dump(data, f)

	def write_tif(self, path, name, scan):
		parent = Path(path)
		folder = parent / f'{name}'
		folder.mkdir(parents=True, exist_ok=True)

		i = 0
		for img in scan: # for each slice
			filename = folder / f'{name}_{str(i).zfill(4)}.tiff'
			if img.size == 0: raise Exception(f'cropped image is empty at fish: {name} slice: {i+1}')

			tiff.imwrite(str(filename), img)
			i += 1
			print(f'[Fish {name}, slice:{i}/{len(scan)}]', end="\r")
		scan = None
		gc.collect()

	def write_clean(self, n, cropped_cts, metadata=None):
		order = self.fish_order_nums[n]
		print(f'order {len(order)}, number of circles: {len(cropped_cts)}')
		print(order)
		if len(order) != len(cropped_cts): raise Exception('Not all/too many fish cropped')
		mastersheet = pd.read_csv('./uCT_mastersheet.csv')

		print(f'[CTFishPy] Writing cropped CT scans {order}')
		for o in range(0, len(order)): # for each fish of number o
			path = Path(f'../../Data/HDD/uCT/low_res_clean/{str(order[o]).zfill(3)}/')

			if not path.exists() : path.mkdir()

			tifpath = path / 'reconstructed_tifs/'
			metapath = path / 'metadata.json'
			if not tifpath.exists() : tifpath.mkdir()

			ct = cropped_cts[o]
			fish = mastersheet.loc[mastersheet['n'] == 100].to_dict()
			weird_fix = list(fish['age'].keys())[0]
			
			if metadata:

				input_metadata = {
					'number'        : order[o],
					'Skip'          : fish['skip'][weird_fix],
					'Age'           : fish['age'][weird_fix],
					'Genotype'      : fish['genotype'][weird_fix],
					'Strain'        : fish['strain'][weird_fix],
					'Name'          : fish['name'][weird_fix],
					'VoxelSizeX'    : metadata['x_voxel_size'],
					'VoxelSizeY'    : metadata['y_voxel_size'],
					'VoxelSizeZ'    : metadata['z_voxel_size'],
					'Comments'      : fish['name'][weird_fix],
					'Phantom'       : fish['name'][weird_fix],
					'Scaling Value' : fish['name'][weird_fix],
					'Arb Value'     : fish['name'][weird_fix]
				}

				self.write_metadata(order[o], input_metadata)

			i = 0
			for img in ct: # for each slice
				filename = tifpath / f'{str(order[o]).zfill(3)}_{str(i).zfill(4)}.tiff'
				if img.size == 0: raise Exception(f'cropped image is empty at fish: {o+1} slice: {i+1}')
				ret = True
				tiff.imwrite(str(filename), img)
				if not ret: raise Exception('image not saved, directory doesnt exist')
				i = i + 1
				print(f'[Fish {order[o]}, slice:{i}/{len(ct)}]', end="\r")
			ct = None
			gc.collect()

	def write_label(self, labelPath, label):
		hf = h5py.File(labelPath, 'w')
		hf.create_dataset('t0', data=label)
		hf.close()
		print('Labels ready.')
		return label

	def to8bit(self, array):
		"""
		Change array from 16bit to 8bit by mapping the data range to 0 - 255

		*NOTE* This does not convert to 8bit normally and is for GUI functions
		"""
		if array.dtype == "uint16":
			new_array = ((array - array.min()) / (array.ptp() / 255.0)).astype(np.uint8)
			return new_array
		else:
			print("image already 8 bit!")
			return new_array

	def detectTubes(self, scan):

		scan = self.to8bit(scan)
		scan = np.array([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in scan])
		m = {'og': scan}

		viewer = napari.Viewer(title='tubeDetector')
		layer = viewer.add_image(scan, metadata=m)

		viewer.window.add_dock_widget(tubeDetector, name="tubeDetector")
		viewer.layers.events.changed.connect(tubeDetector.reset_choices)

		napari.run()
		metadata = layer.metadata
		
		# QTimer().singleShot(500, app.quit)

		return metadata['circle_dict']

	def labelOrder(self, circle_dict):
		scan = circle_dict['labelled_stack']
		ordered_circles = []
		m = {'og': scan, 'circles': circle_dict['circles'], 'ordered_circles' : ordered_circles}

		viewer = napari.Viewer()
		layer = viewer.add_image(scan, metadata=m, name='scan')

		create_orderLabeller(viewer, layer)

		# widgets are stored in napari._qt.widgets.qt_viewer_dock_widget
		napari.run()

		metadata = layer.metadata
		ordered_circles = metadata['ordered_circles']
		return ordered_circles

	def spin(self, scan):
		# create max projection
		projection = np.max(scan, axis=0)
		m = {'og': projection, 'center_rotation' : None, 'angle' : 0}
		viewer = napari.Viewer()
		layer = viewer.add_image(projection, metadata=m, name='projection')

		create_spinner(viewer, layer)
		napari.run()
		metadata = layer.metadata

		angle = metadata['angle']
		center = metadata['center_rotation']
		
		return angle, center

	def lengthMeasure(self, projection):
		return

