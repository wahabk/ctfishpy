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
	new_master = pd.read_csv(new_master_path)
	out_dir = Path('/home/wahab/Data/Local/uCT/dicoms')

	flipped = [223,262,295,501,543]
	test = [40, 223, 41, 200]

	# import pdb; pdb.set_trace()

	for fish in ctreader.fish_nums:
		print(f"Old name: {fish}")
		new_row = new_master.loc[new_master['old_n'] == fish]
		new_name = int(new_row['n'])
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
		new_row['angle'] = scan_metadata['angle']
		new_row['center'] = [scan_metadata['center']]
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
		cv2.imwrite(f'/home/wahab/Data/Local/uCT/projections/z/z_{new_name}.png', z)
		cv2.imwrite(f'/home/wahab/Data/Local/uCT/projections/y/y_{new_name}.png', y)
		cv2.imwrite(f'/home/wahab/Data/Local/uCT/projections/x/x_{new_name}.png', x)

		out_path = out_dir / f"ak_{new_name}.dcm"
		print(out_path)
		ctreader.write_dicom(out_path, new_name, scan)

		# check
		# new_scan = ctreader.read_dicom(out_path)
		# ctreader.view(new_scan)







