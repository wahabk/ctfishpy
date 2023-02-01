import ctfishpy
import napari
import pandas as pd
import numpy as np
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage import io
from skimage.color import rgb2gray
import scipy

if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	# import pdb; pdb.set_trace()

	kitty_path = "examples/Data/tequila_kitty.jpg"

	kitty = io.imread(kitty_path)
	kitty = rgb2gray(kitty)

	kitty = kitty [300:1300, 300:900]

	print(kitty.shape)

	identity = 	[[1,1,1],
				[1,1,1],
				[1,1,1],]

	h_edg = 	[[1,2,1],
				[0,0,0],
				[-1,-2,-1],]

	v_edg = 	[[1,0,-1],
				[2,0,-2],
				[1,0,-1],]
	


	filters = {
		"identity" :	[[1,1,1],
						[1,1,1],
						[1,1,1],],
		"h_edg" : 		[[1,2,1],
						[0,0,0],
						[-1,-2,-1]],
		"v_edg" :	 	[[1,0,-1],
						[2,0,-2],
						[1,0,-1],],	
	}

	for k, f in filters.items():
		convolved_kitty = convolve(kitty, f, mode='same')
		print(k)

		# plt.imshow(convolved_kitty, cmap = "gray")
		# plt.show()
		plt.imsave(f"output/figs/intro/{k}.png" , convolved_kitty, cmap="gray")

	convolved_kitty = ndimage.gaussian_filter(kitty, sigma=(7,7))

	plt.imsave(f"output/figs/intro/gauss.png" , convolved_kitty, cmap="gray")