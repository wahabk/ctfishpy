import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import Trainer, test, CTDataset, agingDataset, precache_age
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
		roiSize = (512,2000),
		train_data = train_data,
		val_data = val_data,
		test_data = test_data,
		batch_size = config['batch_size'],
		bone = bone,
		# n_blocks = config['n_blocks'],
		# norm = config['norm'],
		loss_function = config['loss_function'],
		lr = config['lr'],
		epochs = config['epochs'],
		# start_filters = config['start_filters'],
		# activation = config['activation'],
		num_workers = num_workers,
		n_classes = 33, #including background
		random_seed = 42,
		dropout = config['dropout'],
		n_dims = n_dims
	)
	run['Tags'] = name

	transforms = tio.Compose([
		tio.RandomFlip(axes=(0,1), flip_probability=0.25),
		# tio.RandomAffine(),
		tio.RandomAnisotropy(axes=(0,1),p=0.2),              # make images look anisotropic 25% of times
		tio.RandomBlur(p=0.3),
		tio.OneOf({
			tio.RandomNoise(0.1, 0.01): 0.1,
			tio.RandomBiasField(0.1): 0.1,
			tio.RandomGamma((-0.3,0.3)): 0.1,
			tio.RandomMotion(): 0.3,
		}),
		tio.RescaleIntensity((0.05,0.95)),
	])

	label_size = params['roiSize']

	# create a training data loader
	train_dataset = precache_age(params['dataset_path'], params['n_dims'], params['train_data'], params['bone'], params['roiSize'])
	train_ds = agingDataset(params['dataset_path'], params['bone'], params['train_data'], dataset=train_dataset, roi_size=params['roiSize'], n_classes=params['n_classes'], n_dims=params['n_dims'], transform=transforms)
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)
	# create a validation data loader
	val_dataset = precache_age(params['dataset_path'], params['n_dims'], params['val_data'], params['bone'], params['roiSize'])
	val_ds = agingDataset(params['dataset_path'], params['bone'], params['val_data'], dataset=val_dataset, roi_size=params['roiSize'], n_dims=params['n_dims'], n_classes=params['n_classes'], transform=transforms)
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)

	check_dataset = precache_age(params['dataset_path'], params['n_dims'], params['val_data'], params['bone'], params['roiSize'])
	check_ds = agingDataset(params['dataset_path'], params['bone'], params['val_data'], dataset=check_dataset, roi_size=params['roiSize'], n_dims=params['n_dims'], n_classes=params['n_classes'], transform=transforms)
	check_loader = torch.utils.data.DataLoader(check_ds, batch_size=1, shuffle=False, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)

	print(np.array(check_dataset).shape)
	print(np.array(check_dataset[0]).shape)
	for x,y in check_loader:
		print('checking dataset')
		x = np.array(x[0][0])
		print(x.shape, y)
		plt.imsave("output/tests/aging_dataset.png", x)
		break


	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	model = resnet18(
		# weights=ResNet18_Weights.IMAGENET1K_V1,
		# num_classes=1#params['n_classes'],
	)

	# model = torch.nn.DataParallel(model, device_ids=device_ids)
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
	
	# TODO write test function for aging
	# losses = test(dataset_path, model, test_data, params, threshold=0.5, run=run, criterion=criterion, device=device, num_workers=num_workers, label_size=label_size)
	# run['test/df'].upload(File.as_html(losses))

	run.stop()



if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/uCT'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	config = {
		"lr": 3e-3,
		"batch_size": 8,
		"n_blocks": 3,
		"norm": 'BATCH',
		"epochs": 100,
		"start_filters": 32,
		"activation": "RELU",
		"dropout": 0,
		"loss_function": torch.nn.CrossEntropyLoss(), #k monai.losses.DiceLoss(include_background=False,) #monai.losses.TverskyLoss(include_background=True, alpha=0.7) # # #torch.nn.CrossEntropyLoss()  #  torch.nn.BCEWithLogitsLoss() #BinaryFocalLoss(alpha=1.5, gamma=0.5),
	}
	
	all_data = master[master['age'].notna()]
	# all_data = all_data.drop(index=260)
	# all_data = all_data.drop(index=202)
	print(all_data)
	all_data = all_data.index.to_list()
	# random.shuffle(all_data)

	train_data = all_data[1:400]
	val_data = all_data[401:420]
	test_data =	all_data[420:430]
	print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'test aging'
	save = False
	n_dims = 2
	bone = None
	# save = 'output/weights/aging_resnet.pt'

	work_dir = Path().parent.resolve()

	train_aging(dataset_path, config, name, bone=bone, train_data=train_data, val_data=val_data, n_dims=n_dims,
			test_data=test_data, save=save, tuner=False, device_ids=[0,], num_workers=10, work_dir=work_dir)




