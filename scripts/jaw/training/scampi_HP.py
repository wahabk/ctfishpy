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
from ray.tune.schedulers import ASHAScheduler
from functools import partial

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

from scampi import train

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
	train_data = ready[:25]
	val_data = ready[25:28]
	test_data = ready[25:]
	# train_data = ready[:2]
	# val_data = ready[2:3]
	# test_data = ready[2:3]
	print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'jaw grid search'
	save = False
	# save = 'output/weights/3dunet221019.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'
	model=None

	num_samples = 12
	max_num_epochs = 150
	gpus_per_trial = 1
	device_ids = [0,]
	save = False

	config = {
		"lr": tune.loguniform(1e-5,1e-1),
		"batch_size": tune.choice([4,16,32]),
		"n_blocks": tune.choice([2,3]),
		"norm": tune.choice(["BATCH"]),
		"epochs": 100,
		"start_filters": tune.choice([32]),
		"activation": tune.choice(["RELU"]),
		"dropout": tune.choice([0,0.1]),
		"loss_function": tune.grid_search([
			monai.losses.TverskyLoss(include_background=True, alpha=0.2),
			monai.losses.TverskyLoss(include_background=True, alpha=0.5),
			monai.losses.TverskyLoss(include_background=True, alpha=0.8),
			# monai.losses.GeneralizedDiceLoss(include_background=True),
			# monai.losses.DiceLoss(include_background=True),
			# torch.nn.CrossEntropyLoss(),
			])
	}

	# the scheduler will terminate badly performing trials
	scheduler = ASHAScheduler(
		metric="val_loss",
		mode="min",
		max_t=max_num_epochs,
		grace_period=25,
		reduction_factor=2)
	scheduler = None

	work_dir = Path().parent.resolve()

	result = tune.run(
		partial(train, dataset_path=dataset_path, name=name, bone=bone, train_data=train_data, val_data=val_data, 
			test_data=test_data, save=save, tuner=True, device_ids=[0,], num_workers=10, work_dir=work_dir, dataset_name=dataset_name),
		resources_per_trial={"cpu": 16, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=scheduler,
		checkpoint_at_end=False,
		local_dir=dataset_path+'/RAY_RESULTS/') # Path().parent.resolve()/'ray_results'