import ctfishpy
from ctfishpy.train_utils import undo_one_hot, CTDatasetPredict
import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch
import monai


def predictpatches(model, patch_size, subjects_list, criterion, threshold=0.5):
	"""
	helper function for testing
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	predict_dict = {}
	model.eval()
	
	for idx, subject in enumerate(subjects_list):

		grid_sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=(8,8,8), padding_mode='mean')
		patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
		aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

		with torch.no_grad():
			for i, patch_batch in tqdm(enumerate(patch_loader)):
				input_ = patch_batch['ct'][tio.DATA]
				target = patch_batch['label'][tio.DATA]
				locations = patch_batch[tio.LOCATION]

				input_, target = input_.to(device), target.to(device)

				out = model(input_)  # send through model/network
				out = torch.softmax(out, 1)
				loss = criterion(out, target)
				loss = loss.cpu().numpy()

				# post process to numpy array


				aggregator.add_batch(out, locations)
		output_tensor = aggregator.get_output_tensor()

		array = subject['ct'][tio.DATA].cpu().numpy()
		array = np.squeeze(array)
		y = subject['label'][tio.DATA].cpu()
		y_pred = output_tensor.cpu()  # send to cpu and transform to numpy.ndarray
		predict_dict[idx] = {
			'array': array,
			'y': y, # zero for batch
			'y_pred': y_pred,
			'loss': loss, 
		}

	# TODO just function that returns label
	return predict_dict



if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'

	weights_path = 'output/weights/jaw_unet_230124.pt'

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = ctfishpy.JAW
	hdf5_name = 'jaw_unet_230124'

	done = ctreader.get_hdf5_keys(f"{dataset_path}LABELS/Otolith_unet/Otolith_unet.h5")
	nums = ctreader.fish_nums
	missing = list(set(nums) - set(done))

	model = monai.networks.nets.AttentionUnet(
		spatial_dims=params['spatial_dims'],
		in_channels=1,
		out_channels=params['n_classes'],
		channels=channels,
		strides=strides,
		dropout=params["dropout"],
		# padding='valid',
	)

	label = predictpatches(model, patch_size, subjects_list, criterion)


	



