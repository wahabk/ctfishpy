import ctfishpy
from ctfishpy.bones import Otolith
from ctfishpy.train_utils import predict_oto, CTDatasetPredict
import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch

def old_predict(array, model=None, weights_path=None, threshold=0.5):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	# model
	if model is None:
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			channels=[32, 64, 128],
			strides=[2,2],
			# act=params['activation'],
			# norm=params["norm"],
			padding='valid',
		)

	model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model

	model = model.to(device)
	array = np.array(array/array.max(), dtype=np.float32) # normalise input
	array = np.expand_dims(array, 0) # add batch axis
	array = np.expand_dims(array, 0) # add batch axis
	input_tensor = torch.from_numpy(array)

	print(input_tensor.shape)

	model.eval()
	with torch.no_grad():
		input_tensor.to(device)
		out = model(input_tensor)  # send through model/network
		out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits

	result = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

	label = np.zeros_like(result, dtype='uint8')
	label[result>threshold] = 1
	label[result<threshold] = 0

	return label


if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT'

	weights_path = 'output/weights/3dunet221019.pt'

	ctreader = ctfishpy.CTreader(dataset_path)

	nums = ctreader.fish_nums[:3]

	all_preds = predict_oto(dataset_path=dataset_path, weights_path=weights_path, nums=nums)

	for i, label in enumerate(all_preds):
		num = nums[i]

		ctreader.write_label(bone="Otolith_unet", label = label, n = num, )
