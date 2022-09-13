import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib2 import Path
import napari
import cv2

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/uCT/'
	# dataset_path = '/home/wahab/Data/HDD/uCT'

	ctreader = ctfishpy.CTreader(dataset_path)

	master = ctreader.master
	centers = ctreader.manual_centers
	# print(centers)
	print(len(centers))

	master['otolith_center'] = None
	master = master.astype(object)

	for index, row in master.iterrows():
		old_n = row.old_n
		center = centers[str(old_n)]
		print(index, old_n)

		master.loc[index,'otolith_center'] = np.array(center)

	print(master)
	master.to_csv(ctreader.master_path)