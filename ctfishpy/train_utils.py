"""
Colloidoscope train_utils

This file contains:
- Pytorch datasets
- Pytorch Trainer
- training and testing functions
- training utilities
"""

import ctfishpy
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import math
import os
import copy

import torch
import torch.nn.functional as F
import neptune.new as neptune
from neptune.new.types import File
from ray import tune
import torchio as tio
from .CTreader import CTreader
from .models.unet import UNet
import monai

from sklearn.metrics import roc_curve, auc, roc_auc_score
import gc
import cv2

"""
Datasets
"""

class CTSubjectDataset(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for bones in 3D

	transform is augmentation function

	"""	

	def __init__(self, dataset_path, bone:str, indices:list, roi_size:tuple, n_classes:int, 
				transform=None, label_size:tuple=None, dataset_name=None,
				dataset:np.ndarray=None, labels:np.ndarray=None,  precached:bool=False):
		super().__init__()
		self.dataset_path = dataset_path
		self.bone = bone
		self.indices = indices
		self.roi_size = roi_size
		self.n_classes = n_classes
		self.transform = transform
		self.label_size = label_size
		self.precached = precached
		self.dataset_name=dataset_name
		if self.precached:
			self.dataset = dataset
			self.labels = labels
			assert(len(dataset) == len(labels))

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		ctreader = ctfishpy.CTreader(self.dataset_path)
		# Select sample
		i = self.indices[index] #index is order from precache, i is number from dataset
		# print(index, i)
		
		if self.precached:
			X = self.dataset[index]
			y = self.labels[index]

		else:
			X = ctreader.read(i)
			y = ctreader.read_label(self.bone, i, name=self.dataset_name)
			if self.bone == ctreader.OTOLITHS: center = ctreader.otolith_centers[i]
			if self.bone == ctreader.JAW: center = ctreader.jaw_centers[i]
			# if label size is smaller for roi
			# if self.label_size is not None:
			# 	self.label_size == self.roi_size
			X = ctreader.crop3d(X, self.roi_size, center=center)
			y = ctreader.crop3d(y, self.roi_size, center=center)

		X = np.array(X/X.max(), dtype=np.float32)
		# for reshaping
		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor
		X = torch.from_numpy(X)
		y = torch.from_numpy(y)

		fish = tio.Subject(
			ct=tio.ScalarImage(tensor=X),
			label=tio.LabelMap(tensor=y),
		)

		if self.transform:
			fish = self.transform(fish)

		X = fish.ct.tensor
		y = fish.label.tensor
		y = F.one_hot(y.to(torch.int64), self.n_classes)
		y = y.permute([0,4,1,2,3]) # permute one_hot to channels first after batch
		y = y.squeeze().to(torch.float32)

		fish = tio.Subject(
			ct=tio.ScalarImage(tensor=X),
			label=tio.LabelMap(tensor=y),
		)

		return fish

class CTDataset(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for bones in 3D

	transform is augmentation function

	"""	

	def __init__(self, dataset_path, bone:str, indices:list, roi_size:tuple, n_classes:int, 
				transform=None, label_size:tuple=None, dataset_name=None,
				dataset:np.ndarray=None, labels:np.ndarray=None,  precached:bool=False):
		super().__init__()
		self.dataset_path = dataset_path
		self.bone = bone
		self.indices = indices
		self.roi_size = roi_size
		self.n_classes = n_classes
		self.transform = transform
		self.label_size = label_size
		self.precached = precached
		self.dataset_name=dataset_name
		if self.precached:
			self.dataset = dataset
			self.labels = labels
			assert(len(dataset) == len(labels))

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		ctreader = ctfishpy.CTreader(self.dataset_path)
		# Select sample
		i = self.indices[index] #index is order from precache, i is number from dataset
		# print(index, i)
		
		if self.precached:
			X = self.dataset[index]
			y = self.labels[index]

		else:
			X = ctreader.read(i)
			y = ctreader.read_label(self.bone, i, name=self.dataset_name)
			if self.bone == ctreader.OTOLITHS: center = ctreader.otolith_centers[i]
			if self.bone == ctreader.JAW: center = ctreader.jaw_centers[i]
			# if label size is smaller for roi
			# if self.label_size is not None:
			# 	self.label_size == self.roi_size
			X = ctreader.crop3d(X, self.roi_size, center=center)			
			y = ctreader.crop3d(y, self.roi_size, center=center)

		X = np.array(X/X.max(), dtype=np.float32)
		# for reshaping
		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor
		X = torch.from_numpy(X)
		y = torch.from_numpy(y)

		fish = tio.Subject(
			ct=tio.ScalarImage(tensor=X),
			label=tio.LabelMap(tensor=y),
		)

		if self.transform:
			fish = self.transform(fish)

		X = fish.ct.tensor
		y = fish.label.tensor
		y = F.one_hot(y.to(torch.int64), self.n_classes)
		y = y.permute([0,4,1,2,3]) # permute one_hot to channels first after batch
		y = y.squeeze().to(torch.float32)

		return X, y,

class CTDatasetPredict(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for bones in 3D, for prediction and not training so this doesnt return labels 

	transform is augmentation function

	WARNING test time augmentation is not available

	"""	

	def __init__(self, dataset_path, bone:str, indices:list, roi_size:tuple, n_classes:int, 
				transform=None, dataset:np.ndarray=None,  precached:bool=False):
		super().__init__()
		self.dataset_path = dataset_path
		self.bone = bone
		self.indices = indices
		self.roi_size = roi_size
		self.n_classes = n_classes
		self.transform = transform
		self.precached = precached
		if self.precached:
			self.dataset = dataset

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		ctreader = ctfishpy.CTreader(self.dataset_path)
		# Select sample
		i = self.indices[index] #index is order from precache, i is number from dataset
		# print(index, i)
		
		if self.precached:
			X = self.dataset[index]

		else:
			X = ctreader.read(i)
			if self.bone == "OTOLITH":
				center = ctreader.otolith_centers[i]
			elif self.bone == "JAW":
				raise ValueError("JAW NOT READY")
			X = ctreader.crop3d(X, self.roi_size, center=center)			

		X = np.array(X/X.max(), dtype=np.float32)
		X = np.expand_dims(X, 0)      # if numpy array
		X = torch.from_numpy(X)

		fish = tio.Subject(
			ct=tio.ScalarImage(tensor=X),
		)

		if self.transform:
			raise ValueError("TEST TIME AUG NOT READY")
			fish = self.transform(fish)

		X = fish.ct.tensor

		return X


class CTDataset2D(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for bones in 2D

	transform is augmentation function

	"""	

	def __init__(self, dataset_path, bone:str, indices:list, roi_size:tuple, n_classes:int, 
				transform=None, label_size:tuple=None,
				dataset:np.ndarray=None, labels:np.ndarray=None,  precached:bool=False):
		super().__init__()
		self.dataset_path = dataset_path
		self.bone = bone
		self.indices = indices
		self.roi_size = roi_size
		self.n_classes = n_classes
		self.transform = transform
		self.label_size = label_size
		self.precached = precached
		if self.precached:
			self.dataset = dataset
			self.labels = labels
			assert(len(dataset) == len(labels))

	def __len__(self):
		return len(self.indices)*self.roi_size[0]

	def __getitem__(self, index):

		ctreader = ctfishpy.CTreader(self.dataset_path)
		fish_index, slice_ = self._get_fish(index)

		i = self.indices[fish_index] #index is order from precache, i is number from dataset
		# print(index, i, fish_index, slice_)

		if self.precached:
			X = self.dataset[fish_index]
			y = self.labels[fish_index]

		else:
			X = ctreader.read(i)
			y = ctreader.read_label(self.bone, i)
			center = ctreader.otolith_centers[i]
			# if self.label_size is not None:
			# 	self.label_size == self.roi_size
			X = ctreader.crop3d(X, self.roi_size, center=center)
			y = ctreader.crop3d(y, self.roi_size, center=center)
			
		X = np.array(X/X.max(), dtype=np.float32)
		X = X[slice_]
		y = y[slice_]


		if self.transform:
			transformed = self.transform(image=X, mask=y)
			X = transformed['image']
			y = transformed['mask']

		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor
		X = torch.from_numpy(X)
		y = torch.from_numpy(y)


		y = F.one_hot(y.to(torch.int64), self.n_classes)
		y = y.permute([0,3,1,2]) # permute one_hot to channels first after batch
		y = y.squeeze().to(torch.float32)
		# X = X.unsqueeze(0)
		# y = y.unsqueeze(0)

		
		return X, y,

	def _get_fish(self, index:int):
		"""Get fish to read and slice from pytorch DataLoader index

		Args:
			index (int): Pytorch DataLoader index

		Returns:
			fish_num (int): number of fish from dataset indices
			slice_ (int): slice from the fish to read
		"""

		roiZ = self.roi_size[0]
		n_fish = len(self.indices)
		total_dataset_length = self.__len__()

		fish_num = math.floor(((index)/total_dataset_length)*n_fish)
		slice_ = index - (fish_num * roiZ)

		return fish_num, slice_





class agingDataset(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for aging

	of either the whole scan or roi (e.g.) otoliths

	can be 2d or 3D

	transform is augmentation function

	always from precache

	"""	

	def __init__(self, dataset_path, bone:str, indices:list, roi_size:tuple,
	dataset:list=None, n_classes:int=1, n_dims:int=3, transform=None):
		super().__init__()
		self.dataset_path = dataset_path
		self.bone = bone
		self.indices = indices
		self.roi_size = roi_size
		self.transform = transform
		self.n_classes = n_classes
		self.n_dims = n_dims
		self.dataset = dataset

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		ctreader = ctfishpy.CTreader(self.dataset_path)
		master = ctreader.master
		# Select sample
		i = self.indices[index]-1 #index is order from precache, i is number from dataset
		# print(f'reading index {index}, i is {i}')
		
		X = self.dataset[index]	

		minimum_age = 3
		try:
			age = int(master.iloc[i]['age'])
		except:
			raise ValueError(f"fish {i} has age {master.iloc[i]['age']}")
		# print(f"X_shape {X.shape} age = {age}")
		

		X = np.array(X/X.max(), dtype=np.float32)
		#for reshaping
		X = torch.from_numpy(X)
		X = X.permute(2,0,1)
		X = torch.unsqueeze(X, 3)      # if numpy array
		# print("before transforms x shape", X.shape)

		if self.transform:
			X = self.transform(X)

		y = torch.tensor(age-minimum_age, dtype=torch.int64)
		# y = F.one_hot(y, num_classes=self.n_classes)
		# y = y.to(torch.int64)
		X = torch.squeeze(X, 3)

		return X, y,


def precache(dataset_path, indices, bone, roiSize, label_size=None, dataset_name=None,):

	if label_size is None:
		label_size = roiSize

	ctreader = ctfishpy.CTreader(dataset_path)
	dataset = []
	labels = []
	print("caching...")
	for i in tqdm(indices):
		if bone == ctreader.OTOLITHS: center = ctreader.otolith_centers[i]
		if bone == ctreader.JAW: center = ctreader.jaw_centers[i]

		# roi = ctreader.read_roi(i, roiSize, center)
		scan = ctreader.read(i)
		roi = ctreader.crop3d(scan, roiSize=roiSize, center=center)
		del scan
		dataset.append(roi)

		label = ctreader.read_label(bone, i, name=dataset_name)
		roi_label = ctreader.crop3d(label, roiSize=label_size, center = center)
		labels.append(roi_label)
		del label
		gc.collect()
		# print(roi.shape, roi_label.shape, roi.max(), roi_label.max())
		
	return dataset, labels

def precacheSubjects(dataset_path, indices, bone, roiSize, label_size=None, dataset_name=None,):
	#TODO add transforms

	if label_size is None:
		label_size = roiSize

	ctreader = ctfishpy.CTreader(dataset_path)
	subjects_list = []
	print("caching...")
	for i in tqdm(indices):
		if bone == ctreader.OTOLITHS: center = ctreader.otolith_centers[i]
		if bone == ctreader.JAW: center = ctreader.jaw_centers[i]

		# roi = ctreader.read_roi(i, roiSize, center)
		scan = ctreader.read(i)
		X = ctreader.crop3d(scan, roiSize=roiSize, center=center)
		del scan


		label = ctreader.read_label(bone, i, name=dataset_name)
		y = ctreader.crop3d(label, roiSize=label_size, center = center)
		del label

		gc.collect()

		X = np.array(X/X.max(), dtype="float32")
		X = torch.from_numpy(X)
		y = torch.from_numpy(y)
		X = X.unsqueeze(0)
		y = y.unsqueeze(0)

		y = F.one_hot(y.to(torch.int64), y.max()+1)
		y = y.permute([0,4,1,2,3]) # permute one_hot to channels first after batch
		y = y.squeeze().to(torch.float32)

		fish_subject = tio.Subject(
			ct=tio.ScalarImage(tensor=X),
			label=tio.LabelMap(tensor=y),
		)

		subjects_list.append(fish_subject)

	return subjects_list

def precache_age(dataset_path, n_dims, indices, bone, roiSize):

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	dataset = []
	print("caching...")
	for i in tqdm(indices):
		if bone == 'Otoliths':
			center = ctreader.otolith_centers[i]


		# roi = ctreader.read_roi(i, roiSize, center)
		if n_dims == 2:
			projections = ctreader.read_max_projections(i)
			array = projections[2]
			array = cv2.resize(array, roiSize, interpolation=cv2.INTER_AREA)
		elif n_dims == 3:
			scan = ctreader.read(i)
			if bone == 'Otoliths':
				array = ctreader.crop3d(scan, roiSize=roiSize, center=center)
				del scan
			elif bone is None:
				array = scan
		else:
			raise ValueError(f"n_dims is incorrect value, must be either 2 or 3, you have given {n_dims}")

		dataset.append(array)
		gc.collect()
		# print(roi.shape, roi_label.shape, roi.max(), roi_label.max())
		
	return dataset

def compute_max_depth(shape=1920, max_depth=10, print_out=True):
    shapes = []
    shapes.append(shape)
    for level in range(1, max_depth):
        if shape % 2 ** level == 0 and shape / 2 ** level > 1:
            shapes.append(shape / 2 ** level)
            if print_out:
                print(f'Level {level}: {shape / 2 ** level}')
        else:
            if print_out:
                print(f'Max-level: {level - 1}')
            break
    return shapes

"""
Train and test
"""

def plot_auc_roc(fpr, tpr, auc_roc):
	fig, ax = plt.subplots(1,1)
	ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
	ax.plot([0, 1], [0, 1], 'k--')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('Receiver operating characteristic')
	ax.legend(loc="lower right")

	return fig

def undo_one_hot(result, n_classes, threshold=0.5):
	label = np.zeros(result.shape[1:], dtype = 'uint8')
	for i in range(n_classes):
		# print(f'undoing class {i}')
		if len(result.shape) == 4:
			r = result[i, :, :, :,]
		elif len(result.shape) == 3:
			r = result[i, :, :,]
		else:
			raise Warning(f"result shape unknown {result.shape}")
		label[r>threshold] = i
	return label

def predictpatches(model, patch_size, subjects_list, criterion, threshold=0.5):
	"""
	helper function for testing
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	predict_dict = {}
	model.eval()
	
	for idx, subject in enumerate(subjects_list):

		grid_sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=(8,8,8), padding_mode='mean')
		patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
		aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

		with torch.no_grad():
			for i, patch_batch in tqdm(enumerate(patch_loader)):
				input_ = patch_batch['ct'][tio.DATA]
				target = patch_batch['label'][tio.DATA]
				locations = patch_batch[tio.LOCATION]

				input_, target = input_.to(device), target.to(device)

				out = model(input_)  # send through model/network
				out = torch.softmax(out, 1)
				loss = criterion(out, target)
				loss = loss.cpu().numpy()

				# post process to numpy array


				aggregator.add_batch(out, locations)
		output_tensor = aggregator.get_output_tensor()

		array = subject['ct'][tio.DATA].cpu().numpy()
		array = np.squeeze(array)
		y = subject['label'][tio.DATA].cpu()
		y_pred = output_tensor.cpu()  # send to cpu and transform to numpy.ndarray
		predict_dict[idx] = {
			'array': array,
			'y': y, # zero for batch
			'y_pred': y_pred,
			'loss': loss, 
		}

	return predict_dict

def predict3d(model, test_loader, criterion, threshold=0.5):
	"""
	helper function for testing
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	predict_dict = {}
	model.eval()
	with torch.no_grad():
		for idx, batch in enumerate(test_loader):
			x, y = batch
			x, y = x.to(device), y.to(device)

			out = model(x)  # send through model/network
			out = torch.softmax(out, 1)
			loss = criterion(out, y)
			loss = loss.cpu().numpy()

			# post process to numpy array
			array = x.cpu().numpy()[0,0] # 0 batch, 0 class
			y = y.cpu()
			y_pred = out.cpu()  # send to cpu and transform to numpy.ndarray

			predict_dict[idx] = {
				'array': array,
				'y': y[0],
				'y_pred': y_pred[0],
				'loss': loss, 
			}

	return predict_dict

def predict2d(model, test_loader, criterion, threshold=0.5):
	"""
	helper function for testing

	make sure this predicts on a whole fish and seperates them in the dict, what if batch size is 1
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	array = []
	y_array = []
	y_pred = []
	model.eval()
	with torch.no_grad():
		for idx, batch in enumerate(test_loader):
			x, y = batch
			x, y = x.to(device), y.to(device)

			out = model(x)  # send through model/network
			out = torch.softmax(out, 1)
			loss = criterion(out, y)
			loss = loss.cpu().numpy()

			# post process to numpy array
			batch_array = x.cpu().numpy()[:,0] # : batch, 0 class
			batch_y = y.cpu()
			batch_y_pred = out.cpu()  # send to cpu and transform to numpy.ndarray

			array.append(batch_array)
			y_array.append(batch_y)
			y_pred.append(batch_y_pred)
	
	# array = np.array(array)
	array = np.concatenate(array, axis=0)
	y_array = torch.cat(y_array, dim=0)
	y_pred = torch.cat(y_pred, dim=0)

	y_array = y_array.permute([1,0,2,3])
	y_pred = y_pred.permute([1,0,2,3])

	predict_dict = {
		'array': array,
		'y': y_array,
		'y_pred': y_pred,
		'loss': loss, 
	}

	return predict_dict

def test_jaw(dataset_path, model, bone, test_set, params, threshold=0.5, num_workers=10, 
		batch_size=1, criterion=torch.nn.BCEWithLogitsLoss(), run=False, device='cpu', 
		label_size:tuple=None, dataset_name=None):
	roiSize = params['roiSize']
	n_classes = params['n_classes']
	spatial_dims = params['spatial_dims']
	batch_size = params['batch_size']

	ctreader = ctfishpy.CTreader(dataset_path)
	print('Running test, this may take a while...')

	if label_size is None:
		label_size = roiSize

	if spatial_dims == 3:
		# test_ds = CTDataset(dataset_path, bone, test_set, roiSize, n_classes, transform=None, label_size=label_size, dataset_name=dataset_name) 
		# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
		# predict_dict = predict3d(model, test_loader, criterion=criterion, threshold=threshold)
		subjects_list = precacheSubjects(dataset_path, test_set, bone, roiSize, label_size=label_size, dataset_name=dataset_name)
		predict_dict = predictpatches(model, params['patch_size'], subjects_list, criterion=criterion, threshold=threshold)


	losses = []
	for idx in predict_dict.keys():
		array	= predict_dict[idx]['array']
		y		= predict_dict[idx]['y']
		y_pred	= predict_dict[idx]['y_pred']
		loss	= predict_dict[idx]['loss']
		y_numpy = np.squeeze(y.numpy()) # zero is for batch
		y_pred_numpy = np.squeeze(y_pred.numpy())  # remove batch dim and channel dim -> [H, W]
		y = torch.unsqueeze(y, dim=0)
		y_pred = torch.unsqueeze(y_pred, dim=0)

		i = test_set[idx]
		metadata = ctreader.read_metadata(i)

		print(f"NUM CLASSES FOR ROCAUC tensor: {y.shape} {y_pred.shape}")
		print(f"NUM CLASSES FOR ROCAUC VECTOR: {y_numpy.shape} {y_pred_numpy.shape}")
		# true_vector = np.array(true_label, dtype='uint8').flatten()
		# fpr, tpr, _ = roc_curve(true_vector, pred_vector)
		# fig = plot_auc_roc(fpr, tpr, aucroc)
		# run[f'AUC_{i}'].upload(fig)

		aucroc = roc_auc_score(y_numpy.flatten(), y_pred_numpy.flatten(), average='weighted', multi_class='ovo')
		threshed = torch.zeros_like(y_pred)
		threshed[y_pred > threshold] = 1
		threshed[y_pred < threshold] = 0

		dice_score = monai.metrics.compute_meandice(y_pred=threshed, y=y, include_background=False).numpy()
		g_dice_score = monai.metrics.compute_generalized_dice(y_pred=threshed, y=y, include_background=False).numpy()[0]
		iou = monai.metrics.compute_meaniou(y_pred=threshed, y=y, include_background=False).numpy()[0]
		# iou = torch.mean(iou)
		print(dice_score)
		dice_score = dice_score[0]

		true_label = undo_one_hot(y_numpy, n_classes, threshold=threshold)
		pred_label = undo_one_hot(y_pred_numpy, n_classes, threshold=threshold)

		# print(f"final pred shape {pred_label.shape}")
		# test predict on sim
		array_projections = ctreader.make_max_projections(array)
		label_projections = ctreader.make_max_projections(pred_label)
		labelled_projections = ctreader.label_projections(array_projections, label_projections)
		sidebyside = np.concatenate(labelled_projections[0:2], 0)
		run[f'prediction_{idx}'].upload(File.as_image(sidebyside/sidebyside.max()))
		true_label_projections = ctreader.make_max_projections(true_label)
		labelled_projections = ctreader.label_projections(array_projections, true_label_projections)
		sidebyside = np.concatenate(labelled_projections[0:2], 0)
		run[f'true_{idx}'].upload(File.as_image(sidebyside/sidebyside.max()))


		# TODO threshold label for imaging and one hot encode

		m = {
			'bone': bone,
			'idx'	 : i,
			**metadata,
			'loss'	 : float(loss),
			'aucroc' : float(aucroc),
			'dice'		: dice_score.mean(),
			'dice[1]'	: dice_score[0],
			'dice[2]'	: dice_score[1],
			'dice[3]'	: dice_score[2],
			'dice[4]'	: dice_score[3],
			'g_dice' 	: g_dice_score,
			'iou'		: iou.mean(),
			'iou[1]'	: iou[0],
			'iou[2]'	: iou[1],
			'iou[3]'	: iou[2],
			'iou[4]'	: iou[3],}

		losses.append(m)

	losses = pd.DataFrame(losses)
	losses.loc['mean'] = losses.mean()
	losses.loc['std'] = losses.std()
	print(losses)

	# TODO plot losses against class
	# TODO plot data distrib 
	#plot label brightness distrib

	return losses

def test_otoliths(dataset_path, model, bone, test_set, params, threshold=0.5, num_workers=10, 
		batch_size=1, criterion=torch.nn.BCEWithLogitsLoss(), run=False, device='cpu', 
		label_size:tuple=None, dataset_name=None):
	roiSize = params['roiSize']
	n_classes = params['n_classes']
	spatial_dims = params['spatial_dims']
	batch_size = params['batch_size']

	ctreader = ctfishpy.CTreader(dataset_path)
	print('Running test, this may take a while...')

	if label_size is None:
		label_size = roiSize

	if spatial_dims == 3:
		test_ds = CTDataset(dataset_path, bone, test_set, roiSize, n_classes, transform=None, label_size=label_size, dataset_name=dataset_name) 
		test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
		predict_dict = predict3d(model, test_loader, criterion=criterion, threshold=threshold)
	elif spatial_dims == 2:
		predict_dict = {}
		test_dataset, test_labels = precache(dataset_path, test_set, bone, roiSize)
		for idx, j in enumerate(test_set):
			test_ds = CTDataset2D(dataset_path, bone, [j], roiSize, n_classes, transform=None, label_size=label_size, dataset=[test_dataset[idx]], labels=[test_labels[idx]], precached=True) 
			test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
			pred = predict2d(model, test_loader, criterion=criterion, threshold=threshold)
			predict_dict[idx] = pred

	losses = []
	for idx in predict_dict.keys():
		array	= predict_dict[idx]['array']
		y		= predict_dict[idx]['y']
		y_pred	= predict_dict[idx]['y_pred']
		loss	= predict_dict[idx]['loss']
		y_numpy = np.squeeze(y.numpy()) # zero is for batch
		y_pred_numpy = np.squeeze(y_pred.numpy())  # remove batch dim and channel dim -> [H, W]
		y = torch.unsqueeze(y, dim=0)
		y_pred = torch.unsqueeze(y_pred, dim=0)

		i = test_set[idx]
		metadata = ctreader.read_metadata(i)

		print(f"NUM CLASSES FOR ROCAUC tensor: {y.shape} {y_pred.shape}")
		print(f"NUM CLASSES FOR ROCAUC VECTOR: {y_numpy.shape} {y_pred_numpy.shape}")
		# true_vector = np.array(true_label, dtype='uint8').flatten()
		# fpr, tpr, _ = roc_curve(true_vector, pred_vector)
		# fig = plot_auc_roc(fpr, tpr, aucroc)
		# run[f'AUC_{i}'].upload(fig)

		aucroc = roc_auc_score(y_numpy.flatten(), y_pred_numpy.flatten(), average='weighted', multi_class='ovo')
		threshed = torch.zeros_like(y_pred)
		threshed[y_pred > threshold] = 1
		threshed[y_pred < threshold] = 0

		dice_score = monai.metrics.compute_meandice(y_pred=threshed, y=y, include_background=False).numpy()
		g_dice_score = monai.metrics.compute_generalized_dice(y_pred=threshed, y=y, include_background=False).numpy()[0]
		iou = monai.metrics.compute_meaniou(y_pred=threshed, y=y, include_background=False).numpy()[0]
		# iou = torch.mean(iou)
		print(dice_score)
		dice_score = dice_score[0]

		pred_label = undo_one_hot(y_pred_numpy, n_classes, threshold=threshold)

		# print(f"final pred shape {pred_label.shape}")
		# test predict on sim
		array_projections = ctreader.make_max_projections(array)
		label_projections = ctreader.make_max_projections(pred_label)
		labelled_projections = ctreader.label_projections(array_projections, label_projections)
		sidebyside = np.concatenate(labelled_projections[0:2], 0)
		run[f'prediction_{idx}'].upload(File.as_image(sidebyside/sidebyside.max()))

		# TODO threshold label for imaging and one hot encode

		m = {
			'bone': bone,
			'idx'	 : i,
			**metadata,
			'loss'	 : float(loss),
			'aucroc' : float(aucroc),
			'dice'		: dice_score.mean(),
			'dice[L]'	: dice_score[0],
			'dice[S]'	: dice_score[1],
			'dice[U]'	: dice_score[2],
			'g_dice' 	: g_dice_score,
			'iou'		: iou.mean(),
			'iou[L]'	: iou[0],
			'iou[S]'	: iou[1],
			'iou[U]'	: iou[2],
		}

		losses.append(m)

	losses = pd.DataFrame(losses)
	losses.loc['mean'] = losses.mean()
	losses.loc['std'] = losses.std()
	print(losses)

	# TODO plot losses against class
	# TODO plot data distrib 
	#plot label brightness distrib

	return losses

"""
Trainer
"""

class Trainer:
	def __init__(self,
				 model: torch.nn.Module,
				 device: torch.device,
				 criterion: torch.nn.Module,
				 optimizer: torch.optim.Optimizer,
				 training_DataLoader: torch.utils.data.Dataset,
				 validation_DataLoader: torch.utils.data.Dataset = None,
				 lr_scheduler: torch.optim.lr_scheduler = None,
				 epochs: int = 100,
				 n_classes:int = 1,
				 epoch: int = 0,
				 notebook: bool = False,
				 logger=None,
				 tuner=False,
				 final_activation=None,
				 on_subject=False,
				 ):

		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.training_DataLoader = training_DataLoader
		self.validation_DataLoader = validation_DataLoader
		self.device = device
		self.epochs = epochs
		self.n_classes = n_classes
		self.epoch = epoch
		self.notebook = notebook
		self.logger = logger
		self.tuner = tuner
		self.final_activation = final_activation
		self.on_subject=on_subject

		self.training_loss = []
		self.validation_loss = []
		self.learning_rate = []

	def run_trainer(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		progressbar = trange(self.epochs, desc='Progress')
		for i in progressbar:
			"""Epoch counter"""
			self.epoch += 1  # epoch counter

			"""Training block"""
			self._train()

			if self.logger: self.logger['epochs/loss'].log(self.training_loss[-1])

			"""Validation block"""
			if self.validation_DataLoader is not None:
				self._validate()

			if self.logger: self.logger['epochs/val'].log(self.validation_loss[-1])

			if self.tuner:
				with tune.checkpoint_dir(self.epoch) as checkpoint_dir:
					path = os.path.join(checkpoint_dir, "checkpoint")
					torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)


			"""Learning rate scheduler block"""
			if self.lr_scheduler is not None:
				if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
					self.lr_scheduler.step(self.validation_loss[i])  # learning rate scheduler step with validation loss
				else:
					self.lr_scheduler.step()  # learning rate scheduler step
		return self.training_loss, self.validation_loss, self.learning_rate

	def _train(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		self.model.train()  # train mode
		train_losses = []  # accumulate the losses here
		batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
						  leave=False)

		for i, this_batch in batch_iter:
			if self.on_subject:
				x = this_batch['ct'][tio.DATA]
				y = this_batch['label'][tio.DATA]
				input_, target = x.to(self.device), y.to(self.device)
			else:
				x,y = this_batch
				input_, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
			self.optimizer.zero_grad()  # zerograd the parameters
			out = self.model(input_)  # one forward pass

			if isinstance(self.criterion, (torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss)) == False:
				# print("testing final activation", self.n_classes, out.shape, target.shape)
				out = torch.softmax(out, 1) # this breaks aging (classification) model!
				# print(out, target)

			loss = self.criterion(out, target)  # calculate loss
			loss_value = loss.item()
			train_losses.append(loss_value)

			if self.logger: self.logger['train/loss'].log(loss_value)

			loss.backward()  # one backward pass
			self.optimizer.step()  # update the parameters

			batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

		self.training_loss.append(np.mean(train_losses))
		self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

		batch_iter.close()

	def _validate(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		self.model.eval()  # evaluation mode
		valid_losses = []  # accumulate the losses here
		batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
						  leave=False)

		for i, this_batch in batch_iter:
			if self.on_subject:
				x = this_batch['ct'][tio.DATA]
				y = this_batch['label'][tio.DATA]
				input_, target = x.to(self.device), y.to(self.device)
			else:
				x,y = this_batch
				input_, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
			with torch.no_grad():
				out = self.model(input_)
				if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) == False:
					out = torch.softmax(out, 1)
				loss = self.criterion(out, target)  # calculate loss
				loss_value = loss.item()
				valid_losses.append(loss_value)
				if self.logger: self.logger['val/loss'].log(loss_value)

				batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

		self.validation_loss.append(np.mean(valid_losses))
		if self.tuner: tune.report(val_loss=(np.mean(valid_losses)))


		batch_iter.close()


"""
LR finder
"""


class LearningRateFinder:
	"""
	Train a model using different learning rates within a range to find the optimal learning rate.
	"""

	def __init__(self,
				 model: torch.nn.Module,
				 criterion,
				 optimizer,
				 device
				 ):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.loss_history = {}
		self._model_init = model.state_dict()
		self._opt_init = optimizer.state_dict()
		self.device = device

	def fit(self,
			data_loader: torch.utils.data.DataLoader,
			steps=100,
			min_lr=1e-7,
			max_lr=1,
			constant_increment=False,
			):
		"""
		Trains the model for number of steps using varied learning rate and store the statistics
		"""
		self.loss_history = {}
		self.model.train()
		current_lr = min_lr
		steps_counter = 0
		epochs = math.ceil(steps / len(data_loader))

		progressbar = trange(epochs, desc='Progress')
		for epoch in progressbar:
			batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
							  leave=False)

			for i, (x, y) in batch_iter:
				x, y = x.to(self.device), y.to(self.device)
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = current_lr
				self.optimizer.zero_grad()
				out = self.model(x)
				loss = self.criterion(out, y)
				loss.backward()
				self.optimizer.step()
				self.loss_history[current_lr] = loss.item()

				steps_counter += 1
				if steps_counter > steps:
					break

				if constant_increment:
					current_lr += (max_lr - min_lr) / steps
				else:
					current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)


	def plot(self,
			 smoothing=True,
			 clipping=True,
			 smoothing_factor=0.1
			 ):
		"""
		Shows loss vs learning rate(log scale) in a matplotlib plot
		"""
		loss_data = pd.Series(list(self.loss_history.values()))
		lr_list = list(self.loss_history.keys())
		if smoothing:
			loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
			loss_data = loss_data.divide(pd.Series(
				[1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]))  # bias correction
		if clipping:
			loss_data = loss_data[10:-5]
			lr_list = lr_list[10:-5]
		plt.figure()
		plt.plot(lr_list, loss_data)
		plt.xscale('log')
		plt.title('Loss vs Learning rate')
		plt.xlabel('Learning rate (log scale)')
		plt.ylabel('Loss (exponential moving average)')
		plt.savefig('output/learning_rate_finder.png')

	def reset(self):
		"""
		Resets the model and optimizer to its initial state
		"""
		self.model.load_state_dict(self._model_init)
		self.optimizer.load_state_dict(self._opt_init)
		print('Model and optimizer in initial state.')

"""
Utils
"""

def plot_side_by_side(array, label):
	pass


def renormalise(array):
	array = np.squeeze(array)  # remove batch dim and channel dim -> [H, W]
	array = array * 255
	return array

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

