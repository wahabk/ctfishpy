import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib2 import Path
import napari
import cv2

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/home/wahab/Data/HDD/uCT'

	ctreader = ctfishpy.CTreader(dataset_path)
	lump = ctfishpy.Lumpfish()
	master = ctreader.master

	# df = pd.DataFrame(columns=['Length(mm)'])
	df = pd.read_csv('output/results/vert_lengths.csv', index_col=0)
	print(df)

	wildtypes = ctreader.trim(master, 'genotype', ['wt'])
	sixmonth_wildtypes = ctreader.trim(wildtypes, 'age', [6,7])
	# sixmonth_wildtypes = list(sixmonth_wildtypes['n'])

	nums = list(sixmonth_wildtypes.index)
	print("reading these six month wildtypes", nums[10:])

	for n in nums[10:]: #range(456,466): #ctreader.fish_nums[204:]:
		metadata = ctreader.read_metadata(n)
		# scan = ctreader.read(n)
		# print(scan.shape)
		# scan = ctreader.crop3d(scan, (1900,300,300))
		# print(scan.shape)
		# ctreader.view(scan)
		projections = ctreader.read_max_projections(n)
		# projections = ctreader.make_max_projections(scan)

		p = projections[2]

		print("fish n shape", n, p.shape)
		viewer = napari.Viewer(show=False)
		pixel_length = lump.measure_length(viewer, p)

		length = pixel_length * metadata['VoxelSizeY']

		print(metadata['VoxelSizeY'])
		print(f"fish {n} has len {length}")
		
		df.loc[n] = [length]

	print(df)
	# df.to_csv('output/results/vert_lengths2.csv')



	
