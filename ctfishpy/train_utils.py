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

class CTDataset(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for bones in 3D

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
		return len(self.indices)

	def __getitem__(self, index):
		ctreader = ctfishpy.CTreader(self.dataset_path)
		master = ctreader.master
		# Select sample
		i = self.indices[index] #index is order from precache, i is number from dataset
		
		if self.precached:
			X = self.dataset[index]
			y = self.labels[index]

		else:
			X = ctreader.read(i)
			y = ctreader.read_label(self.bone, i)
			center = ctreader.otolith_centers[i]
			X = ctreader.crop3d(X, self.roi_size, center=center)			
			# if label size is smaller for roi
			if self.label_size is not None:
				self.label_size == self.roi_size
			y = ctreader.crop3d(y, self.label_size, center=center)

		X = np.array(X/X.max(), dtype=np.float32)
		#for reshaping
		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor
		X = torch.from_numpy(X)
		y = torch.from_numpy(y)

		fish = tio.Subject(
			ct=tio.ScalarImage(tensor=X),
			label=tio.LabelMap(tensor=y),
		)
		
		# This weird code is for applying the same transforms to x and y
		# if self.transform:
		# 	if self.label_transform:
		# 		stacked = torch.cat([X, y], dim=0) # shape=(2xHxW)
		# 		stacked = self.label_transform(stacked)
		# 		X, y = torch.chunk(stacked, chunks=2, dim=0)
		# 	X = self.transform(X)

		if self.transform:
			fish = self.transform(fish)


		X = fish.ct.tensor
		y = fish.label.tensor
		y = F.one_hot(y.to(torch.int64), self.n_classes)
		y = y.permute([0,4,1,2,3]) # permute one_hot to channels first after batch
		y = y.squeeze().to(torch.float32)

		return X, y,





class CTDataset2D(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for bones in 2D

	transform is augmentation function

	"""	

	def __init__(self, dataset_path, bone:str, indices:list, roi_size:tuple, n_classes:int, 
				transform=None, label_transform=None, label_size:tuple=None,
				dataset:np.ndarray=None, labels:np.ndarray=None,  precached:bool=False):
		super().__init__()
		self.dataset_path = dataset_path
		self.bone = bone
		self.indices = indices
		self.roi_size = roi_size
		self.n_classes = n_classes
		self.transform = transform
		self.label_transform = label_transform
		self.label_size = label_size
		self.precached = precached
		if self.precached:
			self.dataset = dataset
			self.labels = labels
			assert(len(dataset) == len(labels))

	def __len__(self):
		return len(self.indices)*self.roiz

	def __getitem__(self, index):
		ctreader = ctfishpy.CTreader(self.dataset_path)


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


def precache(dataset_path, indices, bone, roiSize, label_size=None):

	if label_size is None:
		label_size = roiSize

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	dataset = []
	labels = []
	print("caching...")
	for i in tqdm(indices):
		center = ctreader.otolith_centers[i]

		# roi = ctreader.read_roi(i, roiSize, center)
		scan = ctreader.read(i)
		roi = ctreader.crop3d(scan, roiSize=roiSize, center=center)
		del scan
		dataset.append(roi)

		label = ctreader.read_label(bone, i)
		roi_label = ctreader.crop3d(label, roiSize=label_size, center = center)
		labels.append(roi_label)
		del label
		gc.collect()
		# print(roi.shape, roi_label.shape, roi.max(), roi_label.max())
		
	return dataset, labels

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
		r = result[i, :, :, :,]
		label[r>threshold] = i
	return label

def predict(array, model=None, weights_path=None, threshold=0.5):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	# model
	if model is None:
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			channels=[32, 64, 128],
			strides=[2,2],
			# act=params['activation'],
			# norm=params["norm"],
			padding='valid',
		)

	model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model

	model = model.to(device)
	array = np.array(array/array.max(), dtype=np.float32) # normalise input
	array = np.expand_dims(array, 0) # add batch axis
	input_tensor = torch.from_numpy(array)

	model.eval()
	with torch.no_grad():
		input_tensor.to(device)
		out = model(input_tensor)  # send through model/network
		out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits

	result = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

	label = np.zeros_like(result)
	label[result>threshold] = 1
	label[result<threshold] = 0

	return label

def test(dataset_path, model, bone, test_set, params, threshold=0.5, num_workers=4, batch_size=1, criterion=torch.nn.BCEWithLogitsLoss(), run=False, device='cpu', label_size:tuple=None):
	roiSize = params['roiSize']
	n_classes = params['n_classes']

	ctreader = ctfishpy.CTreader(dataset_path)
	print('Running test, this may take a while...')


	if label_size is None:
		label_size = roiSize

	# test on real data
	test_ds = CTDataset(dataset_path, bone, test_set, roiSize, n_classes, transform=None, label_size=label_size) 
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

	losses = []
	model.eval()
	with torch.no_grad():
		for idx, batch in enumerate(test_loader):
			i = test_set[idx]
			metadata = ctreader.read_metadata(i)
			x, y = batch
			x, y = x.to(device), y.to(device)
			# print(x.shape, x.max(), x.min())
			# print(y.shape, y.max(), y.min())

			out = model(x)  # send through model/network
			out = torch.softmax(out, 1)
			loss = criterion(out, y)
			loss = loss.cpu().numpy()

			# post process to numpy array
			y_pred = out.cpu()  # send to cpu and transform to numpy.ndarray
			y_pred_numpy = np.squeeze(y_pred.numpy())  # remove batch dim and channel dim -> [H, W]

			array = x.cpu().numpy()[0,0] # 0 batch, 0 class?
			# array = np.array(array, dtype='uint8')
			y = y.cpu()
			y_numpy = y.numpy()[0] # zero is for batch

			print(f"NUM CLASSES FOR ROCAUC VECTOR: {y_numpy.shape} {y_pred_numpy.shape}")
			# true_vector = np.array(true_label, dtype='uint8').flatten()
			# fpr, tpr, _ = roc_curve(true_vector, pred_vector)
			# fig = plot_auc_roc(fpr, tpr, aucroc)
			# run[f'AUC_{i}'].upload(fig)



			aucroc = roc_auc_score(y_numpy.flatten(), y_pred_numpy.flatten(), average='weighted', multi_class='ovo')
			dice_score = monai.metrics.compute_generalized_dice(y_pred=y_pred, y=y, include_background=False)[0].item()
			threshed = y_pred[y_pred < threshold] = 0
			threshed = y_pred[y_pred > threshold] = 1
			iou = monai.metrics.compute_meaniou(y_pred=y_pred, y=y, include_background=False)

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
				'g_dice' : dice_score,
				'iou'	: iou,
			}

			losses.append(m)

	losses = pd.DataFrame(losses)
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

		for i, (x, y) in batch_iter:
			input_, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
			self.optimizer.zero_grad()  # zerograd the parameters
			out = self.model(input_)  # one forward pass

			if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) == False:
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

		for i, (x, y) in batch_iter:
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

