import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import Trainer, test, CTDataset
from ctfishpy.models import UNet

import matplotlib.pyplot as plt
import neptune.new as neptune
import os
import random
import monai
import math
import torchio as tio
from neptune.new.types import File

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

def train(config, name, bone, train_data, val_data, test_data, save=False, tuner=True, device_ids=[0,1], num_workers=10):
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
		roiSize = (128,128,256),
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
		num_res_units = config['num_res_units'],
		num_workers = num_workers,
		n_classes = 4, #including background
		random_seed = 42,
	)

	run['Tags'] = name
	run['parameters'] = params


	transforms_affine = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.8),
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

	# create a training data loader
	train_ds = CTDataset(params['bone'], params['train_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], transform=transforms_img, label_transform=None, label_size=label_size) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)
	# create a validation data loader
	val_ds = CTDataset(params['bone'], params['val_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], label_size=label_size) 
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
		num_res_units=params["num_res_units"],
		act=params['activation'], # TODO try PReLU
		norm=params["norm"],
	)

	model = torch.nn.DataParallel(model, device_ids=device_ids)
	model.to(device)

	# loss function
	criterion = params['loss_function']
	params['loss_function'] = str(params['loss_function'])

	# optimizer
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	optimizer = torch.optim.Adam(model.parameters(), params['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
	# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, cycle_momentum=False)

	# trainer
	trainer = Trainer(model=model,
					device=device,
					criterion=criterion,
					optimizer=optimizer,
					training_DataLoader=train_loader,
					validation_DataLoader=val_loader,
					lr_scheduler=scheduler,
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

	losses = test(model, bone, test_data, params, threshold=0.1, run=run, criterion=criterion, device=device, num_workers=num_workers, label_size=label_size)
	run['test/df'].upload(File.as_html(losses))

	run.stop()

if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1

	ctreader = ctfishpy.CTreader()

	bone = 'Otoliths'

	old_ns = [78, 200, 218, 240, 242, 257, 259, 277, 330, 337, 341, 364, 385, 421, 423, 443, 459, 461, 462, 463, 464] 
	all_data = [39, 64, 74, 96, 98, 113, 115, 133, 186, 193, 197, 220, 241, 275, 276, 295, 311, 313, 314, 315, 316] 
	all_keys = [1, 39, 64, 74, 96, 98, 112, 113, 115, 133, 186, 193, 197, 220, 241, 275, 276, 295, 311, 313, 314, 315, 316, 371, 374, 420, 427]

	print(f"All data: {len(all_data)}")

	random.shuffle(all_data)
	train_data = all_data[1:14]
	val_data = all_data[14:18]
	test_data =	all_data[18:]
	# train_data = all_data[1:2]
	# val_data = all_data[3:4]
	# test_data =	all_data[4:5]
	print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'trying new test'
	save = False
	# save = 'output/weights/unet.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	#TODO ADD label size
	#TODO use seg models pytorch

	config = {
		"lr": 0.0001,
		"batch_size": 4,
		"n_blocks": 3,
		"norm": 'batch',
		"epochs": 100,
		"start_filters": 16,
		"activation": "RELU",
		"dropout": 0.1,
		"num_res_units": 3,
		"loss_function": torch.nn.CrossEntropyLoss() #monai.losses.DiceLoss(include_background=False,) #  torch.nn.BCEWithLogitsLoss() #BinaryFocalLoss(alpha=1.5, gamma=0.5),
		
	}

	# TODO add model in train

	train(config, name, bone=bone, train_data=train_data, val_data=val_data, 
			test_data=test_data, save=save, tuner=False, device_ids=[0,], num_workers=12)