import ctfishpy
import napari
import numpy as np


if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"
	# name = "JAW_manual"
	name  = "JAW_20221208"
	new_name  = "JAW_20230101"
	roiSize = (256, 256, 320)
	keys = ctreader.get_hdf5_keys(dataset_path+f"/LABELS/JAW/{name}.h5")
	print(keys, len(keys))
	
	sophie = []
	sophie_done = [364,274,50,96,183,337,71,72,182,301,164,116,340,241]
	sophie_missing = [230]
	damiano = [131,216,351,39,139,69,133,135,420,441,220,291,401,250,193]

	for index in damiano:
		print(index)

		scan = ctreader.read(index)
		# label = ctreader.read_label(bone, index, name=name)
		label = ctreader.read_label(bone, index, is_tif=True, name = name)

		# order = input("Which order to rearrange?: ")

		new_label = np.zeros_like(label)

		new_label[label == 3] = 1
		new_label[label == 2] = 2
		new_label[label == 5] = 3
		new_label[label == 4] = 4

		# if index == 131:
		center = ctreader.jaw_centers[index]
		scan_roi = ctreader.crop3d(scan, roiSize, center)
		label_roi = ctreader.crop3d(new_label, roiSize, center)
		ctreader.view(scan_roi, label_roi)

		# ctreader.write_label(bone, new_label, index, name=new_name)



