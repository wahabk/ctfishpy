import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import CTDatasetPrecached, Trainer, test, CTDataset
from ctfishpy.models import UNet

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

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

from torchvision.models import resnet18, ResNet18_Weights


def train_aging(dataset_path, config, name, bone, train_data, val_data, n_dims,
			test_data, save=False, tuner=True, device_ids=[0,1], num_workers=10, work_dir="."):
	
	
	os.chdir(work_dir) # change dir because ray tune changes it

	run = neptune.init(
		project="wahabk/aging",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)  # your credentials


	params = dict(
		dataset_path=dataset_path,
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
		n_dims = n_dims
	)
	run['Tags'] = name

	transforms_affine = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.25),
		# tio.RandomAffine(),
	])
	transforms_img = tio.Compose([
		# tio.RandomAnisotropy(p=0.2),              # make images look anisotropic 25% of times
		# tio.RandomBlur(p=0.3),
		# tio.OneOf({
		# 	tio.RandomNoise(0.1, 0.01): 0.1,
		# 	tio.RandomBiasField(0.1): 0.1,
		# 	tio.RandomGamma((-0.3,0.3)): 0.1,
		# 	tio.RandomMotion(): 0.3,
		# }),
		tio.RescaleIntensity((0.05,0.95)),
	])

	label_size = params['roiSize']

	# create a training data loader
	train_ds = CTDatasetPrecached(params['dataset_path'], params['bone'], train_dataset, train_labels, params['train_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], transform=transforms_img, label_transform=None, label_size=label_size) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)
	# create a validation data loader
	val_dataset, val_labels = precache(params['dataset_path'], params['val_data'], params['bone'], params['roiSize'])
	val_ds = CTDatasetPrecached(params['dataset_path'], params['bone'], val_dataset, val_labels, params['val_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], label_size=label_size) 
	# val_ds = CTDataset(params['bone'], params['val_data'], roi_size=params['roiSize'], n_classes=params['n_classes'], label_size=label_size) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)

	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	model = resnet18(
		weights=ResNet18_Weights.IMAGENET1K_V2,
		num_classes=params['n_classes'],
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
	
	losses = test(dataset_path, model, bone, test_data, params, threshold=0.5, run=run, criterion=criterion, device=device, num_workers=num_workers, label_size=label_size)
	run['test/df'].upload(File.as_html(losses))

	run.stop()



if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/uCT'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1

	ctreader = ctfishpy.CTreader(dataset_path)

	config = {
		"lr": 3e-3,
		"batch_size": 2,
		"n_blocks": 3,
		"norm": 'INSTANCE',
		"epochs": 150,
		"start_filters": 32,
		"activation": "PRELU",
		"dropout": 0.1,
		"loss_function": monai.losses.TverskyLoss(include_background=True, alpha=0.9), #k monai.losses.DiceLoss(include_background=False,) #monai.losses.TverskyLoss(include_background=True, alpha=0.7) # # #torch.nn.CrossEntropyLoss()  #  torch.nn.BCEWithLogitsLoss() #BinaryFocalLoss(alpha=1.5, gamma=0.5),
	}
	
	all_data = ctreader.fish_nums
	random.shuffle(all_data)

	train_data = all_data[1:4]
	val_data = all_data[4:6]
	test_data =	all_data[1:4]
	print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'test aging'
	save = False
	n_dims = 2
	# save = 'output/weights/aging_resnet.pt'

	work_dir = Path().parent.resolve()

	train_aging(dataset_path, config, name, bone=bone, train_data=train_data, val_data=val_data, n_dims=n_dims,
			test_data=test_data, save=save, tuner=False, device_ids=[0,], num_workers=10, work_dir=work_dir)




