import ctfishpy
import napari
import numpy as np


if __name__ == "__main__":
	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"
	# name = "JAW_manual"
	name  = "JAW_20221208"
	roiSize = (256, 256, 320)

	sophie = []
	flipped = [50,96,183,337,71,72,182,301,164,116,340,241]
	sophie_done = [364,274]
	sophie_missing = [230]

	for index in flipped:
		print(index)
		# index = 1

		scan = ctreader.read(index)

		# label = ctreader.read_label(bone, index, name=name)
		label = ctreader.read_label(bone, index, is_tif=True, name = name)

		new_label = np.zeros_like(label)

		new_label[label == 1] = 2
		new_label[label == 2] = 1
		new_label[label == 3] = 4
		new_label[label == 4] = 3

		if index == 50:
			center = ctreader.jaw_centers[index]
			scan_roi = ctreader.crop3d(scan, roiSize, center)
			label_roi = ctreader.crop3d(new_label, roiSize, center)
			ctreader.view(scan_roi, label_roi)

		ctreader.write_label(bone, new_label, index, name=name)



