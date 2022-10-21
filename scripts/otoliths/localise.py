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
import napari

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/uCT/'
	# dataset_path = '/mnt/scratch/ak18001/uCT/'

	weights_path = 'output/weights/3dunet221019.pt'

	ctreader = ctfishpy.CTreader(dataset_path)

	n = 277

	projections = ctreader.read_max_projections(n)

	# viewer = napari.Viewer()
	center = ctreader.localise(projections=projections)

	print(center)

	scan = ctreader.read(n)
	otos = ctreader.crop3d(scan, (128,128,160), center=center)

	ctreader.view(otos)

