import ctfishpy

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch




if __name__ == '__main__':
	# dataset_path = '/home/wahab/Data/HDD/uCT/'
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'

	weights_path = 'output/weights/3dunet221019.pt'

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	missing = list(master.loc[:320].index)

	data_path = "output/results/3d_unet_data20221020.csv"
	new_data_path = "output/results/3d_unet_data20221024.csv"
	df = pd.read_csv(data_path, index_col=0)
	print(df)

	print(missing)

	data_dict = {}
	n_classes = 4
	for n in missing:
		print(n)
		metadata = ctreader.read_metadata(n)
		scan = ctreader.read(n)
		scan = ctreader.crop3d(scan, roiSize=(128,128,160), center=ctreader.otolith_centers[n])
		label = ctreader.read_label("Otolith_unet", n)

		dens = ctreader.getDens(scan, label, n_classes)
		vols = ctreader.getVol(label, metadata, n_classes)

		print(dens, vols)

		data_dict[n] = {
			"Dens1" : dens[0],
			"Dens2" : dens[1],
			"Dens3" : dens[2],
			"Vol1" : vols[0],
			"Vol2" : vols[1],
			"Vol3" : vols[2],
		}

	new_df = pd.DataFrame.from_dict(data_dict, orient='index')
	print(new_df)

	final = pd.concat([new_df, df])

	print(final)
	final.to_csv(new_data_path)








