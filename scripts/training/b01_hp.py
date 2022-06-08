import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import CTDatasetPrecached, Trainer, test, CTDataset
from ctfishpy.models import UNet
import torchio as tio
from neptune.new.types import File
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
from ray import tune
import random
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from pathlib2 import Path

import copy
import monai
import math
from monai.networks.layers.factories import Act, Norm
import gc
from tqdm import tqdm

print(os.cpu_count())
print(torch.cuda.is_available())
print ('Current cuda device ', torch.cuda.current_device())
print('------------num available devices:', torch.cuda.device_count())

import torch.nn as nn
import torch.nn.functional as F 


def dice_loss(true, pred, eps=1e-7):
	"""Computes the Sørensen-Dice loss.
	Note that PyTorch optimizers minimize a loss. In this
	case, we would like to maximize the dice loss so we
	return the negated dice loss.
	Args:
		true: a tensor of shape [B, 1, H, W].
		logits: a tensor of shape [B, C, H, W]. Corresponds to
			the raw output or logits of the model.
		eps: added to the denominator for numerical stability.
	Returns:
		dice_loss: the Sørensen–Dice loss.
	"""

	true = true.type(pred.type())
	dims = (0,) + tuple(range(2, true.ndimension()))
	intersection = torch.sum(pred * true, dims)
	cardinality = torch.sum(pred + true, dims)
	dice_loss = (2. * intersection / (cardinality + eps)).mean()
	return (1 - dice_loss)

def precache(indices, bone, roiSize, label_size=None):

	if label_size is None:
		label_size = roiSize

	ctreader = ctfishpy.CTreader()
	master = ctreader.master
	dataset = []
	labels = []
	print("caching...")
	for i in tqdm(indices):
		old_name = master.iloc[i-1]['old_n']
		center = ctreader.manual_centers[str(old_name)]

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

def train(config, name, bone, train_data, val_data, test_data, save=False, tuner=True, device_ids=[0,1], num_workers=10, work_dir="."):
	os.chdir(work_dir)
	'''
	by default for ray tune
	'''

	# setup neptune
	run = neptune.init(
		project="wahabk/Fishnet",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)

	params = dict(
		bone=bone,
		roiSize = (128,128,160),
		train_data = train_data,
		val_data = val_data,
		test_data = test_data,
		batch_size = config['batch_size'],
		n_blocks = config['n_blocks'],
		norm = config['norm'],
		loss_function = config['loss_function'],
		lr = config['lr'],
		epochs = config['epochs'],
		start_filters = config['start_filters'],
		activation = config['activation'],
		num_workers = num_workers,
		n_classes = 4, #including background
		random_seed = 42,
		dropout = config['dropout'],
	)

	run['Tags'] = name
	


	transforms_affine = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.25),
		tio.RandomAffine(),
	])
	transforms_img = tio.Compose([
		tio.RandomAnisotropy(p=0.2),              # make images look anisotropic 25% of times
		tio.RandomBlur(p=0.3),
		tio.OneOf({
			tio.RandomNoise(0.1, 0.01): 0.1,
			tio.RandomBiasField(0.1): 0.1,
			tio.RandomGamma((-0.3,0.3)): 0.1,
			tio.RandomMotion(): 0.3,
		}),
		tio.RescaleIntensity((0.05,0.95)),
	])

	#TODO find a way to precalculate this - should i only unpad the first block?
	# if config['n_blocks'] == 2: label_size = (48,48,48)
	# if config['n_blocks'] == 3: label_size = (24,24,24)
	label_size = params['roiSize']

	train_dataset, train_labels = precache(params['train_data'], params['bone'], params['roiSize'])
	print(train_dataset[0].shape, train_dataset[0].max())
	# create a training data loader
	train_ds = CTDatasetPrecached(params['bone'], train_dataset, train_labels, params['train_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], transform=transforms_img, label_transform=None, label_size=label_size) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)
	# create a validation data loader
	val_dataset, val_labels = precache(params['val_data'], params['bone'], params['roiSize'])
	val_ds = CTDatasetPrecached(params['bone'], val_dataset, val_labels, params['val_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], label_size=label_size) 
	# val_ds = CTDataset(params['bone'], params['val_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], label_size=label_size) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)

	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	start_filters = params['start_filters']
	n_blocks = params['n_blocks']
	start = int(math.sqrt(start_filters))
	channels = [2**n for n in range(start, start + n_blocks)]
	strides = [2 for n in range(1, n_blocks)]

	# model
	model = monai.networks.nets.UNet(
		spatial_dims=3,
		in_channels=1,
		out_channels=params['n_classes'],
		channels=channels,
		strides=strides,
		num_res_units=params["n_blocks"],
		act=params['activation'], # TODO try PReLU
		norm=params["norm"],
		dropout=params["dropout"],
	)

	model = torch.nn.DataParallel(model, device_ids=device_ids)
	model.to(device)

	# loss function
	criterion = params['loss_function']
	if isinstance(criterion, monai.losses.TverskyLoss):
		params['alpha'] = criterion.alpha
	params['loss_function'] = str(params['loss_function'])
	run['parameters'] = params

	# optimizer
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	optimizer = torch.optim.Adam(model.parameters(), params['lr'])
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
	# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, cycle_momentum=False)

	# trainer
	trainer = Trainer(model=model,
					device=device,
					criterion=criterion,
					optimizer=optimizer,
					training_DataLoader=train_loader,
					validation_DataLoader=val_loader,
					# lr_scheduler=scheduler,
					epochs=params['epochs'],
					n_classes=params['n_classes'],
					logger=run,
					tuner=tuner,
					)

	# start training
	training_losses, validation_losses, lr_rates = trainer.run_trainer()

	run['learning_rates'].log(lr_rates)
	
	if save:
		model_name = save
		torch.save(model.state_dict(), model_name)
		# run['model/weights'].upload(model_name)

	train_dataset, train_labels = None, None
	val_dataset, val_labels = None, None

	gc.collect()
	losses = test(model, bone, test_data, params, threshold=0.5, run=run, criterion=criterion, device=device, num_workers=num_workers, label_size=label_size)
	run['test/df'].upload(File.as_html(losses))

	run.stop()



if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dataset_path = "/home/ak18001/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = 'Otoliths'

	old_ns = 	[40, 78, 200, 218, 240, 242, 256, 257, 259, 277, 330, 337, 341, 364, 385, 421, 423, 443, 459, 461, 462, 463, 464, 527, 530, 582, 589]
	all_data = 	[39, 64, 74, 96, 98, 113, 115, 133, 186, 193, 197, 220, 241, 275, 276, 295, 311, 313, 314, 315, 316] 
	all_keys = 	[1, 39, 64, 74, 96, 98, 112, 113, 115, 133, 186, 193, 197, 220, 241, 275, 276, 295, 311, 313, 314, 315, 316, 371, 374, 420, 427]
	test_data = [1,64,374,427]

	print(f"All data: {len(all_keys)}")
	[all_keys.remove(i) for i in test_data]

	random.shuffle(all_keys)
	train_data = all_keys[1:20]
	val_data = all_keys[20:]
	# test_data =	all_keys[22:]
	# train_data = all_data[1:4]
	# val_data = all_data[4:6]
	# test_data =	all_data[1:4]
	print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'HP SAUCE with FLAV'
	save = False
	# save = 'output/weights/unet.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	#TODO ADD label size

	num_samples = 50
	max_num_epochs = 400
	gpus_per_trial = 1
	device_ids = [0,]
	save = False

	config = {
		"lr": 3e-3,#tune.loguniform(0.01, 0.00001),
		"batch_size": tune.choice([1,2,4]),
		"n_blocks": tune.randint(2,4),
		"norm": tune.choice(["INSTANCE"]),
		"epochs": 150,
		"start_filters": tune.choice([8,32]),
		"activation": tune.choice(["PRELU"]),
		"dropout": tune.choice([0,0.1]),
		"loss_function": tune.grid_search([monai.losses.TverskyLoss(include_background=True, alpha=0.1), 
											monai.losses.TverskyLoss(include_background=True, alpha=0.2),
											monai.losses.TverskyLoss(include_background=True, alpha=0.3),
											monai.losses.TverskyLoss(include_background=True, alpha=0.4),
											monai.losses.TverskyLoss(include_background=True, alpha=0.5),
											monai.losses.TverskyLoss(include_background=True, alpha=0.6),
											monai.losses.TverskyLoss(include_background=True, alpha=0.7),
											monai.losses.TverskyLoss(include_background=True, alpha=0.8),
											monai.losses.TverskyLoss(include_background=True, alpha=0.9),])#  ,monai.losses.DiceLoss(include_background=False,), , torch.nn.CrossEntropyLoss()]) #BinaryFocalLoss(alpha=1.5, gamma=0.5), 
	}

	# the scheduler will terminate badly performing trials
	# scheduler = ASHAScheduler(
	# 	metric="val_loss",
	# 	mode="min",
	# 	max_t=max_num_epochs,
	# 	grace_period=1,
	# 	reduction_factor=2)

	work_dir = Path().parent.resolve()

	result = tune.run(
		partial(train, name=name, bone=bone, train_data=train_data, val_data=val_data, 
			test_data=test_data, save=save, tuner=True, device_ids=[0,], num_workers=10, work_dir=work_dir),
		resources_per_trial={"cpu": 10, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=None,
		checkpoint_at_end=False,
		local_dir='/home/ak18001/Data/HDD/uCT/RAY_RESULTS') # Path().parent.resolve()/'ray_results'
