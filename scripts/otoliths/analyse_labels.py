import ctfishpy

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch




if __name__ == '__main__':
	dataset_path = '/home/wahab/Data/HDD/uCT/'
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	# dataset_path = '/mnt/scratch/ak18001/uCT/'

	weights_path = 'output/weights/3dunet221019.pt'

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	broken = np.array([276, 277, 278, 279, 280, 318, 319, 320])



    