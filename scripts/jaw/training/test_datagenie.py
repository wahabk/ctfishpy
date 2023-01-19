import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import Trainer, test_jaw, precacheSubjects, CTSubjectDataset

import matplotlib.pyplot as plt
import neptune.new as neptune
import os
import random
import monai
import math
import torchio as tio
from neptune.new.types import File
from tqdm import tqdm
import gc
import torch.nn.functional as F
from pathlib2 import Path
from ray import tune

def renormalise(array):
	array = np.squeeze(array)  # remove batch dim and channel dim -> [H, W]
	array = array * 255
	return array

def undo_one_hot(result, n_classes, threshold=0.5):
	label = np.zeros(result.shape[1:], dtype = 'uint8')
	for i in range(n_classes):
		if len(result.shape) == 4:
			r = result[i, :, :, :,]
		elif len(result.shape) == 3:
			r = result[i, :, :,]
		else:
			raise Warning(f"result shape unknown {result.shape}")
		label[r>threshold] = i
	return label

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/uCT'
	# dataset_path = '/mnt/scratch/ak18001/uCT'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	curated = [257,351,241,164,50,39,116,441,291,193,420,274,364,401,72,71,69,250,182,183,301,108,216,340,139,337,220,1,154,230,131,133,135,96,98,]
	damiano = [131,216,351,39,139,69,133,135,420,441,220,291,401,250,193]
	ready = [1, 50, 71, 72, 96, 116, 164, 182, 183, 241, 257, 274, 301, 337, 340, 364]+damiano
	bone = ctfishpy.JAW
	dataset_name = "JAW_20230101"

	keys = ctreader.get_hdf5_keys(f"{dataset_path}/LABELS/{bone}/{dataset_name}.h5")
	print(f"all keys len {len(keys)} nums {keys}")

	remove = [216,257,274] # 216 hi res, 257 bad seg from me, 274 sp7 fucked
	ready = [x for x in ready if x not in remove]
	print(f"All data: {len(ready)}, nums  {ready}")

	random.seed(42)
	random.shuffle(ready)
	# train_data = ready[:25]
	# val_data = ready[25:28]
	# test_data = ready[25:]
	train_data = ready[:2]
	val_data = ready[2:3]
	test_data = ready[2:3]
	print(f"train = {train_data} val = {val_data} test = {test_data}")

	num_workers = 10

	config = {
		"lr": 9e-3,
		"batch_size": 16,
		"n_blocks":4,
		"norm": 'BATCH',
		"epochs": 200,
		"start_filters": 32,
		"activation": "RELU",
		"dropout": 0.1,
		"loss_function": monai.losses.TverskyLoss(include_background=False, alpha=0.5), 
	}


	params = dict(
		dataset_path=dataset_path,
		bone=bone,
		dataset_name=dataset_name,
		roiSize = (200, 192, 256),
		patch_size = (64,64,64),
		sampler_probs = {0:3, 1:4, 2:4, 3:5, 4:5},
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
		n_classes = 5, #including background
		random_seed = 42,
		dropout = config['dropout'],
		spatial_dims = 3,
	)
	
	transforms = tio.Compose([
		tio.RandomFlip(axes=(0), flip_probability=0.5),
		tio.RandomAffine(p=0.5),
		tio.RandomBlur(p=0.2),
		tio.RandomBiasField(0.75, order=4, p=0.5),
		tio.RandomNoise(1, 0.02, p=0.5),
		tio.RandomGamma((-0.3,0.3), p=0.25),
		tio.ZNormalization(p=0.5),
		tio.RescaleIntensity(percentiles=(1,99), p=1),
	])

	#TODO find a way to precalculate this for tiling
	# if config['n_blocks'] == 2: label_size = (48,48,48)
	# if config['n_blocks'] == 3: label_size = (24,24,24)
	label_size = params['roiSize']

	train_subjects = precacheSubjects(params['dataset_path'], params['train_data'], params['bone'], params['roiSize'], dataset_name=params['dataset_name'])
	train_ds = tio.SubjectsDataset(train_subjects, transform=transforms) 
	patch_sampler = tio.LabelSampler(params['patch_size'], 'label', params['sampler_probs'])
	patches_queue = tio.Queue(
		train_ds,
		max_length=800,
		samples_per_volume=40,
		sampler=patch_sampler,
		num_workers=params['num_workers'],
	)
	train_loader = torch.utils.data.DataLoader(patches_queue, batch_size=params['batch_size'], shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())


	for batch in train_loader:
		print(batch.keys())
		xs,ys = batch['ct'][tio.DATA], batch['label'][tio.DATA]

		for x,y in zip(xs, ys):
			print(x.shape, y.shape)

			x = np.squeeze(np.array(x))*255
			y = undo_one_hot(np.array(y), n_classes=5)

			ctreader.view(x, label=y)



