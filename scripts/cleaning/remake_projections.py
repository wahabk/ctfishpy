import ctfishpy
import numpy as np
from tifffile import imsave


if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"
	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	for  n  in ctreader.fish_nums:
		scan =  ctreader.read(n)

		projections = ctreader.make_max_projections(scan)

		out_path = f"/home/ak18001/Data/HDD/uCT/PROJECTIONS/NEW_TIFS/Z/Z_{n}.tif"
		imsave(out_path, projections[0])
		out_path = f"/home/ak18001/Data/HDD/uCT/PROJECTIONS/NEW_TIFS/X/X_{n}.tif"
		imsave(out_path, projections[1])
		out_path = f"/home/ak18001/Data/HDD/uCT/PROJECTIONS/NEW_TIFS/Y/Y_{n}.tif"
		imsave(out_path, projections[2])
