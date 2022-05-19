"""
Saving dicoms

make write dicom in lumpfish

read_clean
rotate align
flip
rename ak_n
make projections
save dicom
make 8bit and 16 bit dataset?
save dicoms all together or all sep?
"""

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

	new_master_path = "ctfishpy/Metadata/uCT_mastersheet_test.csv"
	new_master = pd.read_csv(new_master_path, index_col='n')
	# out_dir = Path('/home/wahab/Data/Local/uCT/dicoms')
	out_dir = Path('/home/ak18001/Data/HDD/uCT/DICOMS')

	with open(ctreader.anglePath, "r") as fp:
		angles = json.load(fp)


	print(new_master)

	flipped = [223,262,295,501,543]
	test = [40, 468, 223, 41, 200, ]
	skip = [405]

	print(ctreader.fish_nums[262:])
	# exit()

	for fish in ctreader.fish_nums[262:]:
		print(f"Old name: {fish}")
		new_row = new_master.loc[new_master['old_n'] == fish]
		# import pdb; pdb.set_trace()
		new_name = int(new_row.index[0])
		
		print(f"New name: {new_name}")

		#read and align
		scan, scan_metadata = ctreader.read(fish, align=True)
		print("scan_metadata", scan_metadata)
		# exit()

		#flip if needed
		if fish in flipped:
			scan = scan [::-1]

		# save metadata to new mastersheet
		new_row['shape'] = [scan.shape]
		new_row['size'] = scan.size
		new_row['angle'] = angles[str(fish)]['angle'] # scan_metadata['angle']
		new_row['center'] = [angles[str(fish)]['center']] #  [scan_metadata['center']]
		new_row['VoxelSizeX'] = scan_metadata['VoxelSizeX']
		new_row['VoxelSizeY'] = scan_metadata['VoxelSizeY']
		new_row['VoxelSizeZ'] = scan_metadata['VoxelSizeZ']
		new_row['Phantom'] = scan_metadata['Phantom']
		new_row['Arb Value'] = scan_metadata['Arb Value']
		new_row['Scaling Value'] = scan_metadata['Scaling Value']
		new_master.loc[new_master['old_n'] == fish] = new_row
		new_master.to_csv(new_master_path)
		# print(new_master)

		z, y, x = ctreader.make_max_projections(scan)
		cv2.imwrite(f'/home/ak18001/Data/HDD/uCT/PROJECTIONS/Z/z_{new_name}.png', z)
		cv2.imwrite(f'/home/ak18001/Data/HDD/uCT/PROJECTIONS/Y/y_{new_name}.png', y)
		cv2.imwrite(f'/home/ak18001/Data/HDD/uCT/PROJECTIONS/X/x_{new_name}.png', x)
		xy = np.concatenate([x,y], axis=1)
		cv2.imwrite(f'/home/ak18001/Data/HDD/uCT/PROJECTIONS/XY/xy_{new_name}.png', xy)

		out_path = out_dir / f"ak_{new_name}.dcm"
		print(out_path)
		ctreader.write_dicom(out_path, new_name, scan)

		# check
		# new_scan = ctreader.read_dicom(out_path)
		# ctreader.view(new_scan)







