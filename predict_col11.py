import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import random
import gc
from pathlib2 import Path
import json

if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	unet = ctfishpy.Unet('Otoliths')
	unet.weightsname = 'final2d'
	nums = ctreader.fish_nums
	# random.shuffle(nums)
	nclasses = 3
	data = {}

	dataset_path = Path('/home/ak18001/Data/HDD/uCT/Misc/yushi_data/n')

	jsonpath = dataset_path / 'centers.json'
	with open(jsonpath, 'r') as fp:
		centers = json.load(fp)

	for fish in dataset_path.iterdir():
		# if str(fish) not in done:
		if fish.is_dir():
			print(fish)

			# ct = ctreader.read_path(fish)
			center = centers[str(fish.stem)]
			print(center)

			stack_metadata={
				'VoxelSizeX' : 0.0202360326938826,
				'VoxelSizeY' : 0.0202360326938826,
				'VoxelSizeZ' : 0.0202360326938826,
				}


			label, ct = unet.predict_custom(fish, center, thresh=0.5)

			print(label.shape, ct.shape)
			roiSize = (128,128,256)
			label = ctreader.crop3d(label, roiSize, center)
			ct = ctreader.crop3d(ct, roiSize, center)
			print(label.shape, ct.shape)
			ctreader.make_gif(ct, f'output/test_pred{fish.stem}.gif', fps=30, label = label)

			densities = ctreader.getDens(ct, label, nclasses)
			volumes = ctreader.getVol(label, stack_metadata, nclasses)

			data[fish.stem] = {'densities': list(densities), 'vols': list(volumes)}


	datapath = "output/col11_new_data.json"
	with open(datapath, 'w') as f:
			json.dump(data, f, sort_keys=True, indent=4)