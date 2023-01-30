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
import torchio as tio
from tqdm import tqdm


def jaw_predict(array:np.ndarray, model=None, patch_size=(160,160,160), patch_overlap=(16,16,16), weights_path=None, threshold=0.5):
	"""
	jaw predict 
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	jaw_nclasses = 5

	if model is None:
		start_filters = 32
		n_blocks = 5
		start = int(math.sqrt(start_filters))
		channels = [2**n for n in range(start, start + n_blocks)]
		strides = [2 for _ in range(1, n_blocks)]
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=3,
			in_channels=1,
			out_channels=jaw_nclasses,
			channels=channels,
			strides=strides,
		)

	model = torch.nn.DataParallel(model, device_ids=None)
	model.to(device)

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model

	# The weights require dataparallel because it's used in training
	# But dataparallel doesn't work on cpu so remove it if need be
	if device == "cpu": model = model.module.to(device)
	elif device == "cuda": model = model.to(device)

	array = np.array(array/array.max(), dtype=np.float32)
	array = np.expand_dims(array, 0)
	X = torch.from_numpy(array)

	subject = tio.Subject(
		ct=tio.ScalarImage(tensor=X),
	)

	grid_sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap, padding_mode='mean')
	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
	aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

	with torch.no_grad():
		for i, patch_batch in tqdm(enumerate(patch_loader)):
			input_ = patch_batch['ct'][tio.DATA]
			locations = patch_batch[tio.LOCATION]

			input_= input_.to(device)

			out = model(input_)  # send through model/network
			out = torch.softmax(out, 1)

			aggregator.add_batch(out, locations)

	output_tensor = aggregator.get_output_tensor()

	# post process to numpy array
	label = output_tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	label = undo_one_hot(label, jaw_nclasses, threshold)

	# TODO norm brightness histo?
	return label



if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'
	ctreader = ctfishpy.CTreader(dataset_path)

	weights_path = 'output/weights/jaw_unet_230124.pt'
	jaw_roi = (192,192,192) 
	jaw_nclasses = 5
	bone = ctfishpy.JAW
	hdf5_name = 'jaw_unet_230124'

	# done = ctreader.get_hdf5_keys(f"{dataset_path}LABELS/Otolith_unet/Otolith_unet.h5")
	# nums = ctreader.fish_nums
	# missing = list(set(nums) - set(done))

	data_dict = {}
	for index in ctreader.fish_nums:
		print(index)
		center = ctreader.jaw_centers[index]

		scan = ctreader.read(index)
		scan_roi = ctreader.crop3d(scan, jaw_roi, center)

		label = jaw_predict(scan_roi, weights_path=weights_path)
		print(scan_roi.shape, label.shape)

		metadata = ctreader.read_metadata(index)
		dens = ctreader.getDens(scan_roi, label, jaw_nclasses)
		vols = ctreader.getVol(label, metadata, jaw_nclasses)
		print(dens, vols)

		data_dict[index] = {
			"Dens1" : dens[0],
			"Dens2" : dens[1],
			"Dens3" : dens[2],
			"Dens4" : dens[3],
			"Vol1" : vols[0],
			"Vol2" : vols[1],
			"Vol3" : vols[2],
			"Vol4" : vols[3],
		}

		full_label = ctreader.uncrop3d(scan, label, center)
		ctreader.write_label(ctreader.JAW, full_label, index, hdf5_name)

	print(data_dict)
	df = pd.DataFrame.from_dict(data_dict, orient='index')
	print(df)
	df.to_csv("output/results/jawunet_data230124.csv")







