import torch
import numpy as np
import ctfishpy
from ctfishpy.train_utils import Trainer, test_jaw, CTDataset, precache

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

def train(config, dataset_path, name, bone, train_data, val_data, test_data, model=None, save=False, tuner=True, device_ids=[0,1], num_workers=10, work_dir=".", dataset_name=None):
	os.chdir(work_dir)
	'''
	by default for ray tune
	'''

	# setup neptune
	run = neptune.init_run(
		project="wahabk/Fishnet",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)

	params = dict(
		dataset_path=dataset_path,
		bone=bone,
		dataset_name=dataset_name,
		roiSize = (256, 256, 320),
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

	run['Tags'] = name
	
	transforms = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.25),
		tio.RandomAffine(p=0.25),
		tio.RandomBlur(p=0.3),
		tio.RandomBiasField(0.4, p=0.5),
		tio.RandomNoise(0.1, 0.01, p=0.25),
		tio.RandomGamma((-0.3,0.3), p=0.25),
		tio.ZNormalization(),
		tio.RescaleIntensity(percentiles=(0.5,99.5)),
	])

	#TODO find a way to precalculate this for tiling
	# if config['n_blocks'] == 2: label_size = (48,48,48)
	# if config['n_blocks'] == 3: label_size = (24,24,24)
	label_size = params['roiSize']

	train_dataset, train_labels = precache(params['dataset_path'], params['train_data'], params['bone'], params['roiSize'], dataset_name=params['dataset_name'])
	print(train_dataset[0].shape, train_dataset[0].max())

	train_ds = CTDataset(dataset_path=params['dataset_path'], bone=params['bone'], indices=params['train_data'],
						dataset=train_dataset, labels=train_labels, roi_size=params['roiSize'], n_classes=params['n_classes'], 
						dataset_name=dataset_name, transform=transforms, label_size=label_size, precached=True) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)

	val_dataset, val_labels = precache(params['dataset_path'], params['val_data'], params['bone'], params['roiSize'], dataset_name=params['dataset_name'])
	val_ds = CTDataset(dataset_path=params['dataset_path'], bone=params['bone'], indices=params['val_data'],
					dataset=val_dataset, labels=val_labels, roi_size=params['roiSize'], n_classes=params['n_classes'], 
					transform=None, label_size=label_size, precached=True) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available(), persistent_workers=True)

	# TODO Add label sampler

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	start_filters = params['start_filters']
	n_blocks = params['n_blocks']
	start = int(math.sqrt(start_filters))
	channels = [2**n for n in range(start, start + n_blocks)]
	strides = [2 for _ in range(1, n_blocks)]

	if model is None:
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=params['spatial_dims'],
			in_channels=1,
			out_channels=params['n_classes'],
			channels=channels,
			strides=strides,
			dropout=params["dropout"],
			# padding='valid',
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
	# optimizer = torch.optim.RMSprop(model.parameters(), params['lr'])
	# optimizer = torch.optim.SGD(model.parameters(), params['lr'])
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
					tuner=tuner,)

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
	losses = test_jaw(dataset_path, model, bone, test_data, params, threshold=0.5, 
					run=run, criterion=criterion, device=device, num_workers=num_workers, 
					label_size=label_size, dataset_name=dataset_name)
	run['test/df'].upload(File.as_html(losses))

	run.stop()

if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/uCT'
	dataset_path = '/mnt/scratch/ak18001/uCT'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	curated = [257,351,241,164,50,39,116,441,291,193,420,274,364,401,72,71,69,250,182,183,301,108,216,340,139,337,220,1,154,230,131,133,135,96,98,]
	ready = [1, 50, 71, 72, 96, 116, 164, 182, 183, 241, 257, 274, 301, 337, 340, 364]
	bone = ctfishpy.JAW
	dataset_name = "JAW_20221208"

	keys = ctreader.get_hdf5_keys(f"{dataset_path}/LABELS/{bone}/{dataset_name}.h5")
	print(f"all keys len {len(keys)} nums {keys}")
	print(f"All data: {len(ready)}")

	random.seed(42)
	random.shuffle(ready)
	train_data = ready[:12]
	val_data = ready[12:]
	test_data = ready[12:]
	print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'JAW start'
	save = False
	# save = 'output/weights/3dunet221019.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	config = {
		"lr": 3e-3,
		"batch_size": 1,
		"n_blocks": 2,
		"norm": 'INSTANCE',
		"epochs": 50,
		"start_filters": 32,
		"activation": "RELU",
		"dropout": 0.0,
		"loss_function": monai.losses.TverskyLoss(include_background=True, alpha=0.5), 
	}

	# TODO add model in train?
	work_dir = Path().parent.resolve()

	train(config, dataset_path, name, bone=bone, train_data=train_data, val_data=val_data, 
		test_data=test_data, save=save, tuner=False, device_ids=[0,], num_workers=10, 
		dataset_name=dataset_name ,work_dir=work_dir)

