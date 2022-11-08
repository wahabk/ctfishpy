import ctfishpy
import napari
import numpy as np
import pandas as pd

if __name__ == "__main__":

	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)
	centers_out_path = "output/results/jaw/jaw_centers.csv"
	bone = "JAW"

	# sample = [257, 351, 241, 244, 164, 456, 39, 441, 291, 193, 420, 355, 72, 71, 196, 216, 139, 431, 220, 5, 131, 133, 96, ]

	# df = pd.DataFrame(columns=["jaw_center"])
	df = pd.read_csv(centers_out_path, index_col=0)
	df["jaw_center"] = df["jaw_center"].astype('object')
	for n in ctreader.fish_nums[50:]:
		print(n)
		# scan = ctreader.read(n)
		# projections = ctreader.make_max_projections(scan)
		projections = ctreader.read_max_projections(n)
		# center = ctreader.otolith_centers[n]
		# print(center)
		center = ctreader.localise(projections=projections, to_use=[1,2])
		print(center)

		# otos = ctreader.crop3d(scan, (256,256,320), center=center)
		# print(otos.shape)
		# ctreader.view(otos)

		df.at[n, "jaw_center"] = center
		df.to_csv(centers_out_path)

	df = pd.read_csv(centers_out_path, index_col=0)
	print(df)