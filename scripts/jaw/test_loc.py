import ctfishpy
import napari
import numpy as np
import pandas as pd

if __name__ == "__main__":

	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)
	centers_out_path = "output/results/jaw/jaw_centers.csv"
	bone = "JAW"
	master = ctreader.master
	print(master)
	print(master.otolith_center)
	print(master.jaw_center)

	print(ctreader.jaw_centers[1])

	# jaw_centers = pd.read_csv(centers_out_path, index_col=0, dtype=object)
	# print(jaw_centers)
	# jaw_centers = jaw_centers.jaw_center.to_dict()

	# jaw_dict = {k: np.fromstring(c[1:-1], dtype='uint16', sep=',') for k, c in jaw_centers.items() }

	# print(jaw_dict[1])
	# print(jaw_dict[1][0])

	# df = pd.DataFrame.from_dict(jaw_dict)
	# df.to_csv("output/results/jaw/jaw_centers_test.csv")













