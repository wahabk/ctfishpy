"""
Lumpfish

Lumpfish are cleaner fish added to salmon farms. This script serves to seperate all the data cleaning and wrangling
code from the main body of CTfishpy.

You should not be using this unless you've spoken to the author. Please use CTreader.

"""

from multiprocessing.sharedctypes import Value
from cv2 import sort
from deprecated import deprecated
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
import math

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

	def read_tiff(self, path, r = None, scale = 40):
		tifpath = Path(path)
		files = sorted(tifpath.iterdir())
		images = [str(f) for f in files if f.suffix == '.tif']

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

	def write_tif(self, path, name, scan, metadata=None):
		parent = Path(path)
		folder = parent / f'{name}'
		tifpath = folder / 'reconstructed_tifs'
		tifpath.mkdir(parents=True, exist_ok=True)
		metadata_path = folder / 'metadata.json'

		i = 0
		for img in tqdm(scan): # for each slice
			filename = tifpath / f'{name}_{str(i).zfill(4)}.tiff'
			if img.size == 0: raise Exception(f'cropped image is empty at fish: {name} slice: {i+1}')

			tiff.imwrite(str(filename), img)
			i += 1
		
		if metadata:
			with open(metadata_path, "w") as fp:
				json.dump(metadata, fp=fp, sort_keys=True, indent=4)

		scan = None
		gc.collect()

	@deprecated
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
		hf.create_dataset('0', data=label)
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
			return array

	def detectTubes(self, viewer, scan):

		from .GUI import tubeDetector

		scan = self.to8bit(scan)
		scan = np.array([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in scan])
		m = {'og': scan}

		viewer = napari.Viewer()
		layer = viewer.add_image(scan, metadata=m)

		viewer.window.add_dock_widget(tubeDetector, name="tubeDetector")
		viewer.layers.events.changed.connect(tubeDetector.reset_choices)

		viewer.show(block=True)
		metadata = layer.metadata
		
		return metadata['circle_dict']

	def labelOrder(self, viewer, circle_dict):

		from .GUI import create_orderLabeller

		scan = circle_dict['labelled_stack']
		ordered_circles = []
		m = {'og': scan, 'circles': circle_dict['circles'], 'ordered_circles' : ordered_circles}

		layer = viewer.add_image(scan, metadata=m, name='scan')

		create_orderLabeller(viewer, layer)

		# widgets are stored in napari._qt.widgets.qt_viewer_dock_widget
		viewer.show(block=True)

		metadata = layer.metadata
		ordered_circles = metadata['ordered_circles']
		return ordered_circles

	def spin(self, viewer, scan):

		from .GUI import create_spinner

		# create max projection
		scan = self.to8bit(scan)
		scan = np.array([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in scan])
		projection = np.max(scan, axis=0)
		m = {'og': projection, 'center_rotation' : None, 'angle' : 0}

		layer = viewer.add_image(projection, metadata=m, name='projection')
		create_spinner(viewer, layer)
		viewer.show(block=True)
		# TODO will this remove the need for viewer to be instantiated each time in main thread?
		# viewer.close() 

		metadata = layer.metadata
		angle = metadata['angle']
		center = metadata['center_rotation']
		return angle, center
		
	def measure_length(self, viewer:napari.Viewer, projection:np.ndarray):

		from .GUI import create_fishRuler
		
		meta = {'og': projection, 'head' : 0, 'tail' : 0}
		layer = viewer.add_image(projection, metadata=meta, name='projection')

		create_fishRuler(viewer, layer)
		viewer.show(block=True)

		metadata = layer.metadata
		head = metadata['head']
		tail = metadata['tail']

		if head == 0 or tail == 0:
			raise ValueError(f"No input recieved, pixel locations selected: (head={head}, tail={tail})")

		pixel_length = math.dist(head, tail)

		return pixel_length

