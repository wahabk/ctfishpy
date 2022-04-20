import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import napari
import cv2


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()
	master = ctreader.mastersheet()

	# for i in [40, 468, 200]:
	# 	scan, metadata = ctreader.read(i, align=True)
	# 	ctreader.view(scan)

	scan, metadata = ctreader.read(468, align=False)
	# viewer = napari.Viewer(show=False)
	# angle, center = lump.spin(viewer, scan)
	# print(angle, center)

	# TODO write dicom

	# array = ctreader.read_dicom('examples/Data/image1.dcm')
	# print(array.shape)
	# array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	# ctreader.view(array)

	# save temp metadata with shape as practice







	path = Path("/home/wahab/Data/Local/uCT/low_res_dicoms")
	name = "ak_test"
	print(scan.dtype)
	print(scan.size, scan.itemsize)

	print(path/name)

	ctreader.write_dicom(path, name, scan)

	name = name+".dcm"
	array = ctreader.read_dicom(path/name)
	print(array.shape)
	ctreader.view(array)

