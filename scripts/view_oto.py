import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib2 import Path
import napari
import cv2
import ast

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/home/wahab/Data/HDD/uCT'

	ctreader = ctfishpy.CTreader(dataset_path)


	fish_num = 1
	scan = ctreader.read(fish_num)

	master = ctreader.master
	c = master.loc[fish_num, 'otolith_center']
	print(c)
	center = np.fromstring(c[1:-1], dtype='uint16', sep=' ')
	print(center)
	print(center[0])

	oto = ctreader.crop3d(scan, roiSize=(128,128,128), center=center)
	oto[oto<((140/255)*65535)] = 0
	oto = ctreader.to8bit(oto)


	ctreader.view(oto)
	