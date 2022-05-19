import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy
from copy import deepcopy
import napari
import pandas as pd
import math
import seaborn as sns
from skimage import io

if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()

	lump = ctfishpy.Lumpfish()

	path = '/home/wahab/Data/HDD/uCT/qiao/yushi_data/QT_051_055/QT_051_055_[tifs]/'

	ct = lump.read_tiff(path, r=(500,1500), scale = 40)

	# label = np.zeros(ct.shape, dtype='uint8')232
	# viewer = napari.view_image(ct)
	# viewer.add_labels(label)
	# napari.run()
	# label = viewer.layers['label'].data

	label = io.imread('output/label.tif')

	print(label.shape, label.max(), label.min())

	phantom = np.mean(ct[label==1])
	dens_calib = 0.75 / phantom
	print(dens_calib)

	import pdb; pdb.set_trace()


	

	# ctreader.write_label('Phantoms', label, 1)
