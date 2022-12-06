import ctfishpy
import napari
import numpy as np


if __name__ == "__main__":
	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"
	name = "JAW_manual"
	roiSize = (256, 256, 320)

	index = 50

	scan = ctreader.read(index)

	# label = ctreader.read_label(bone, index, name=name)
	label = ctreader.read_label(bone, index, is_amira=True)


	center = ctreader.jaw_centers[index]
	scan = ctreader.crop3d(scan, roiSize, center)
	label = ctreader.crop3d(label, roiSize, center)


	ctreader.view(scan, label)


