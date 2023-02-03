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

from ray.tune.search.optuna import OptunaSearch
from ray import tune, air
from ray.air import session

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
	bone = ctfishpy.JAW
	dataset_name = "JAW_20230124"

	# curated = [257,351,241,164,50,39,116,441,291,193,420,274,364,401,72,71,69,250,182,183,301,108,216,340,139,337,220,1,154,230,131,133,135,96,98,]
	# damiano = [131,216,351,39,139,69,133,135,420,441,220,291,401,250,193]
	# ready = [1, 50, 71, 72, 96, 116, 164, 182, 183, 241, 257, 274, 301, 337, 340, 364]+damiano

	keys = ctreader.get_hdf5_keys(f"{dataset_path}/LABELS/{bone}/{dataset_name}.h5")
	print(f"all keys len {len(keys)} nums {keys}")

	# remove = [216,] # 216 hi res, 257 bad seg from me, 274 sp7 fucked
	# ready = [x for x in ready if x not in remove]
	# print(f"All data: {len(ready)}, nums  {ready}")

	# random.seed(42)
	# random.shuffle(ready)
	# train_data = ready[:25]
	# val_data = ready[25:28]
	# test_data = ready[25:]
	# print(f"train = {train_data} val = {val_data} test = {test_data}")
	name = 'jaw cv'
	save = False
	# save = 'output/weights/3dunet221019.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'
	model=None

	num_samples = 12
	max_num_epochs = 200
	gpus_per_trial = 1
	device_ids = [0,]
	save = False
	work_dir = Path().parent.resolve()


	config = {
		"lr": 0.00263078,
		"batch_size": 2,
		"n_blocks":6,
		"norm": 'BATCH',
		"epochs": 150,
		"start_filters": 16,
		"kernel_size": 5,
		"activation": "RELU",
		"dropout": 0.0001,
		"patch_size": (192,192,192),
		"loss_function": monai.losses.TverskyLoss(include_background=False, alpha=0.2), 
	}

	curated = [257,351,241,164,50,39,116,441,291,193,420,274,364,401,72,71,69,250,182,183,301,108,216,340,139,337,220,1,154,230,131,133,135,96,98,]
	curated.remove(216)
	curated.remove(108)
	curated.remove(154)
	curated.remove(98)
	# curated.remove(230)

	missing = list(set(keys).difference(curated))
	extra = list(set(curated).difference(keys))
	print(f"difference between curated and keys {missing}{extra}")

	crossval_folds = {
		"young_wt" 			:[241, 50, 39, 164],
		"young_mutants" 	:[257, 351, 441, 116],
		"1yr_wt" 			:[72, 71, 69, 401],
		"1yr_mut" 			:[193, 420, 274, 364],
		"1yr_het" 			:[291, 183, 250, 182],
		"2yr_wt" 			:[220, 1, 340],
		"2yr_mut" 			:[337, 139, 301, 230],
		"3yr_wt" 			:[133, 135, 96, 131],
	}

	cv_array = np.array([
		[241, 50, 39, 164],
		[257, 351, 441, 116],
		[72, 71, 69, 401],
		[193, 420, 274, 364],
		[291, 183, 250, 182],
		[220, 1, 340],
		[337, 139, 301, 230],
		[133, 135, 96, 131],
	], dtype = object)
	flattened_cv_array = np.hstack(cv_array)

	missing = list(set(flattened_cv_array).difference(curated))
	extra = list(set(curated).difference(flattened_cv_array))
	if len(missing) > 0 or len(extra) > 0:
		print(missing, extra)
		raise ValueError("something wrong with folds")

	print(len(curated))

	for i, (k, fold) in enumerate(crossval_folds.items()):
		name = f'jaw cv-{i+1}-{k}'
		print(f"\n\n\n !!! ---- training {name}  ---- !!! \n\n\n")
		print(k, fold)

		train_data = [x for x in curated if x not in fold]
		# train_data = train_data[0]
		val_data = fold
		test_data = fold

		missing = list(set(fold).difference(curated))
		extra = list(set(curated).difference(fold))

		print(missing)
		print(len(test_data), len(train_data))

		work_dir = Path().parent.resolve()
		train(config, dataset_path, name, bone=bone, train_data=train_data, val_data=val_data, model=model, 
			test_data=test_data, save=save, tuner=False, device_ids=[0,], num_workers=16, 
			dataset_name=dataset_name ,work_dir=work_dir)