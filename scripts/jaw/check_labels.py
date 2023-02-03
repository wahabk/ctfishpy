import ctfishpy
import napari
import numpy as np
from scipy import ndimage


if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"
	# name = "JAW_manual"
	# name  = "JAW_20221208"
	name  = "JAW_20230124"
	roiSize = (192, 192, 256)

	keys = ctreader.get_hdf5_keys(f"{dataset_path}/LABELS/{bone}/{dataset_name}.h5")
	print(f"all keys len {len(keys)} nums {keys}")

	for i in keys:
		print(i)
		scan = ctreader.read(i)
		label = ctreader.read_label(bone, i, name=name)
		center = ctreader.jaw_centers[i]
		scan = ctreader.crop3d(scan, roiSize, center=center)
		label = ctreader.crop3d(label, roiSize, center=center)
		ctreader.view(scan, label)


		