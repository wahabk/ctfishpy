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




	#rerotate468
	# scan, metadata = ctreader.read(468, align=False)
	# viewer = napari.Viewer(show=False)
	# angle, center = lump.spin(viewer, scan)
	# print(angle, center)


	# save temp metadata with shape as practice





	# path = Path("/home/wahab/Data/Local/uCT/low_res_dicoms")
	# name = "ak_test"
	# print(scan.dtype)
	# print(scan.size, scan.itemsize)

	# print(path/name)

	# ctreader.write_dicom(path, name, scan)

	# name = name+".dcm"
	# array = ctreader.read_dicom(path/name)
	# print(array.shape)
	# ctreader.view(array)

	out_path = Path('/home/wahab/Data/Local/uCT/low_res_clean')
	bad_bois = [424, 429, 433, 434, 435, 465, 467, 468, 534, 543, 559]
	for fish in bad_bois:

		scan, metadata = ctreader.read(fish)
		ctreader.view(scan)

		new_path = out_path / str(fish) / "reconstructed_tifs"
		scan = ctreader.read_path(new_path)
		ctreader.view(scan)


