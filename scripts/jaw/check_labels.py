import ctfishpy
import napari
import numpy as np
from scipy import ndimage


if __name__ == "__main__":
	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"
	# name = "JAW_manual"
	# name  = "JAW_20221208"
	dataset_name  = "JAW_20230124"
	roiSize = (192, 192, 320)


	crossval_folds = {
		"young_wt" 			:[241, 50, 39, 164],
		"young_mutants" 	:[257, 351, 441, 116],
		"1yr_wt" 			:[72, 71, 69, 401],
		"1yr_mut" 			:[193, 420, 274, 364],
		"1yr_het" 			:[291, 183, 250, 182],
		"2yr_wt" 			:[220, 1, 340],
		"2yr_mut" 			:[337, 139, 301, 230],
		"3yr_wt" 			:[133, 135, 96, 131],
	}


	keys = ctreader.get_hdf5_keys(f"{dataset_path}/LABELS/{bone}/{dataset_name}.h5")
	print(f"all keys len {len(keys)} nums {keys}")

	for k, fold in crossval_folds.items():

		print(f"\n\n{k}\n\n")
		for i in fold:
			print(f"\n\n{i}\n\n")
			scan = ctreader.read(i)
			label = ctreader.read_label(bone, i, name=dataset_name)
			center = ctreader.jaw_centers[i]
			scan = ctreader.crop3d(scan, roiSize, center=center)
			label = ctreader.crop3d(label, roiSize, center=center)
			ctreader.view(scan, label)


		