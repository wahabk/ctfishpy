import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import Trainer, train, test
from ctfishpy.models import UNet

import matplotlib.pyplot as plt
import neptune.new as neptune
import os
import random

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1

	ctreader = ctfishpy.CTreader()

	bone = 'otoliths'
	wahab_samples 	= [78,200,218,240,277,330,337,341,462,464,364,385]
	mariel_samples	= [421,423,242,463,259,459,461]
	zac_samples		= [257,443,218,364,464]

	all_data = wahab_samples+mariel_samples+zac_samples
	print(f"All data: {len(all_data)}")
	random.shuffle(all_data)

	train_data = all_data[0:10]
	val_data = all_data[11:14]
	test_data =	all_data[14:17]
	name = 'replicating otoliths'
	save = 'output/weights/unet.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	#TODO ADD label size
	#TODO use seg models pytorch

	config = {
		"lr": 0.001,
		"batch_size": 4,
		"n_blocks": 2,
		"norm": 'batch',
		"epochs": 20,
		"start_filters": 32,
		"activation": 'relu',
		"loss_function": torch.nn.BCEWithLogitsLoss() #BinaryFocalLoss(alpha=1.5, gamma=0.5),
	}

	# TODO add model in train

	train(config, name, bone=bone, train_data=train_data, val_data=val_data, 
			test_data=test_data, save=save, tuner=False, device_ids=[0,])
