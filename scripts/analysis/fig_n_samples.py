import ctfishpy
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
	dataset_path = "/home/wahab/Data/HDD/uCT"
	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = '/mnt/scratch/ak18001/uCT'
	# dataset_path = None
	ctreader = ctfishpy.CTreader(data_path=dataset_path)
	master = ctreader.master

	data_path = "output/results/n_train_2.csv"

	df = pd.read_csv(data_path)
	print(df)

	x = np.arange(1,19,2)
	y1 = np.array(df["Unet_2D[Dice]"])
	e1 = np.array(df["Unet_2D[Dice]STD"])
	y2 = np.array(df["Unet_3D[Dice]"])
	e2 = np.array(df["Unet_3D[Dice]STD"])

	lower_bound1 = y1 - e1
	upper_bound1 = y1 + e1
	lower_bound2 = y2 - e2
	upper_bound2 = y2 + e2

	print(x)
	print(y1)

	plt.errorbar(x, y1, yerr=e1, fmt="-b")
	plt.fill_between(x, lower_bound1, upper_bound1, alpha=.3, color='b')
	plt.errorbar(x, y2, yerr=e2, fmt="-r")
	plt.fill_between(x, lower_bound2, upper_bound2, alpha=.3, color='r')
	plt.ylim((0,1))

	
	plt.show()