import ctfishpy
import numpy as np
from tifffile import imsave
import imageio

if __name__ == "__main__":
	# dataset_path = "/home/ak18001/Data/HDD/uCT"
	dataset_path = "/home/wahab/Data/HDD/uCT"
	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	index = 1

	bone = ctreader.OTOLITHS
	oto_label = ctreader.read_label(bone, index)

	bone = ctreader.JAW
	jaw_label = ctreader.read_label(bone, index, name = "JAW_20221208")

	combined = jaw_label + oto_label

	scan = ctreader.read(1)

	scan = scan[1000:]
	combined = combined[1000:]

	ctreader.view(scan, label=combined)

	# imageio.mimsave('/path/to/movie.gif', images)
