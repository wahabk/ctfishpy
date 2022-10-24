import ctfishpy
from ctfishpy.train_utils import undo_one_hot, CTDatasetPredict
import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch

def predict_oto(dataset_path, weights_path, nums, model=None):
	"""
	helper function for prediction
	"""

	ctreader = ctfishpy.CTreader(dataset_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	bone = "OTOLITH"
	n_blocks = 3
	start_filters = 32
	roi = (128,128,160)
	n_classes = 4
	batch_size = 2
	num_workers = 10

	start = int(math.sqrt(start_filters))
	channels = [2**n for n in range(start, start + n_blocks)]
	strides = [2 for n in range(1, n_blocks)]

	# model
	if model is None:
		model = model = monai.networks.nets.UNet(
			spatial_dims=3,
			in_channels=1,
			out_channels=4,
			channels=channels,
			strides=strides,
			num_res_units=n_blocks,
			act="PRELU",
			norm="INSTANCE",
		)

	model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model

	pred_loader = CTDatasetPredict(dataset_path, bone=bone, indices=nums, roi_size=roi, n_classes=n_classes)
	pred_loader = torch.utils.data.DataLoader(pred_loader, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

	predict_list = []
	data_dict = {}
	model.eval()
	with torch.no_grad():
		for idx, batch in enumerate(pred_loader):
			x = batch
			x = x.to(device)

			out = model(x)  # send through model/network
			out = torch.softmax(out, 1)

			# post process to numpy array
			array_batch = x.cpu().numpy()[:,0] # : batch, 0 class
			array_batch = array_batch * 65535
			y_pred_batch = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray


			#TODO find phantoms of each and add to mastersheet
			zipped = zip(array_batch, y_pred_batch)
			for i, (array, pred) in enumerate(zipped):
				num = nums[(idx*batch_size)+i]
				y_pred = undo_one_hot(pred, n_classes=n_classes)
				print(f"appending this pred index {num} shape {y_pred.shape}")
				predict_list.append(y_pred)

				ctreader.write_label(bone="Otolith_unet", label = y_pred, n = num)

				metadata = ctreader.read_metadata(num)
				dens = ctreader.getDens(array, y_pred, n_classes)
				vols = ctreader.getVol(y_pred, metadata, n_classes)

				data_dict[num] = {
					"Dens1" : dens[0],
					"Dens2" : dens[1],
					"Dens3" : dens[2],
					"Vol1" : vols[0],
					"Vol2" : vols[1],
					"Vol3" : vols[2],
				}

	# predict_list = np.concatenate(predict_list, axis=0)

	return predict_list, data_dict

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'

	weights_path = 'output/weights/3dunet221019.pt'

	ctreader = ctfishpy.CTreader(dataset_path)

	#TODO localise 

	done = ctreader.get_hdf5_keys(f"{dataset_path}LABELS/Otolith_unet/Otolith_unet.h5")
	nums = ctreader.fish_nums
	missing = list(set(nums) - set(done))
	#broken = [276,277,278,279,280,318,319,320]
	#[nums.remove(i) for i in broken]
	missing.sort()
	print(missing)
	#exit()

	all_preds, data_dict = predict_oto(dataset_path=dataset_path, weights_path=weights_path, nums=missing)

	print(data_dict)

	df = pd.DataFrame.from_dict(data_dict, orient='index')
	print(df)
	# df.to_csv("output/results/3d_unet_data20221020.csv")

