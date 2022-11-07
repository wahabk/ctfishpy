import ctfishpy
import napari
import numpy as np
import pandas as pd

if __name__ == "__main__":

	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	dataset_path = "/home/wahab/Data/HDD/uCT"
	
	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"

	# sample = [257, 351, 241, 244, 164, 456, 39, 441, 291, 193, 420, 355, 72, 71, 196, 216, 139, 431, 220, 5, 131, 133, 96, ]

	centers = pd.DataFrame(columns=["jaw_center"])
	for n in ctreader.fish_nums:
		print(n)
		# scan = ctreader.read(n)
		projections = ctreader.read_max_projections(n)
		# center = ctreader.otolith_centers[n]
		# print(center)
		center = ctreader.localise(projections=projections)


		scan = ctreader.read(n)
		otos = ctreader.crop3d(scan, (256,256,320), center=center)
		ctreader.view(otos)

		centers.loc[n, "jaw_center"] = center
		centers.to_csv("output/results/jaw_centers.csv")