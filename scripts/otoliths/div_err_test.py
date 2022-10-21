import ctfishpy
from ctfishpy.bones import Otolith
from ctfishpy.train_utils import undo_one_hot, CTDatasetPredict
import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'

	weights_path = 'output/weights/3dunet221019.pt'

	ctreader = ctfishpy.CTreader(dataset_path)

	missing = [276, 277, 278, 279, 280, 318, 319, 320]

	for n in missing:
		print(n)
		X = ctreader.read(n)
		loc = ctreader.otolith_centers[n]
		print(loc)
		X = ctreader.crop3d(X, roiSize=(128,128,160), center=loc)
		print(X.shape)
		X = np.array(X/X.max(), dtype=np.float32)

