import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import napari
import cv2

def test_metadata(master:pd.DataFrame, dataset_path:str):
	#TODO see whats in master not in Data
	master.index

	ctreader.fish_nums

	pass

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()
	master = ctreader.mastersheet()

	# scan = ctreader.read_dicom(ctreader.dataset_path / "DICOMS/ak_1.dcm")
	# scan = ctreader.to8bit(scan)
	# ctreader.view(scan)

	missing = [103, 208, 209, 210, 211, 212, 213, 214, 215, 405, 406, 422, 495, 496, 504, 505, 506, 507, 508, 509, 554, 555, 556, 557, 558, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 639]

	print(master)

	old_nums = ctreader.fish_nums

	new_nums = [int(n.stem.split('_')[1]) for n in ctreader.dicoms_path.iterdir()]
	new_nums = sorted(new_nums)

	master = master.replace({np.nan : None})
	shapes = master['shape'].to_list()

	indices = [True if n is None else False for n in shapes] 
	missings = master.iloc[indices]
	exists = []
	for m in missing:
		if m in ctreader.fish_nums:
			exists.append(m)

	print('missing', missing)
	print('exists', exists)

	# rename dicoms after rm
	current_names = master['ak_n'].to_list()
	new_names = master.index.to_list()

	print(current_names)
	print(new_names)
	# exit()

	# rename dicoms
	for old, new in zip(current_names, new_names):
		if old != new:
			path = ctreader.dicoms_path / f"ak_{old}.dcm"
			new_path = ctreader.dicoms_path / f"ak_{new}.dcm"
			print(f"renaming {old} to {new}")
			path.rename(new_path)

	#rename projections
	# for axis in ['X', 'XY', 'Y', 'Z']:
	# 	for old, new in zip(current_names, new_names):
	# 		path = ctreader.dataset_path / "PROJECTIONS" / axis / f"{axis.lower()}_{old}.png"
	# 		new_path = ctreader.dataset_path / "PROJECTIONS" / axis / f"{axis.lower()}_{new}.png"
	# 		print(f"renaming {path} to {new_path}")
	# 		path.rename(new_path)


	#test dixoms
	# import pdb; pdb.set_trace()
	# for i in [40, 468, 200]:
	# 	scan, metadata = ctreader.read(i, align=True)
	# 	ctreader.view(scan)





	#rerotate468
	# scan, metadata = ctreader.read(435, align=False)
	# ctreader.view(scan)
	# viewer = napari.Viewer(show=False)
	# angle, center = lump.spin(viewer, scan)
	# print(angle, center)







	# save temp metadata with shape as practice
	# fixing names and tifpaths
	# path = Path("/home/wahab/Data/Local/uCT/low_res_dicoms")
	# name = "ak_test"
	# print(scan.dtype)
	# print(scan.size, scan.itemsize)

	# print(path/name)

	# ctreader.write_dicom(path, name, scan)

	# name = name+".dcm"
	# array = ctreader.read_dicom(path/name)
	# print(array.shape)
	# ctreader.view(array)

	# out_path = Path('/home/wahab/Data/Local/uCT/low_res_clean')
	# done = []
	# bad_bois = [424, 429, 433, 434, 435,465, 467, 468, 534, 543, 559]
	# for fish in bad_bois:

	# 	scan, metadata = ctreader.read(fish, align=True)
	# 	ctreader.view(scan)

	# 	new_path = out_path / str(fish) / "reconstructed_tifs"
	# 	scan = ctreader.read_path(new_path)
	# 	ctreader.view(scan)








	#fix anglepath
	# with open(ctreader.anglePath, "r") as fp:
	# 	angles = json.load(fp)

	# print(angles)

	# new_angles = {}

	# for fish, angle, in angles.items():
	# 	angles[fish]
	# 	new_angles[fish] = {}
	# 	new_angles[fish]['angle'] = angle
	# 	new_angles[fish]['center'] = None

	# print(new_angles)
	# with open("ctfishpy/Metadata/new_angles.json", "w") as fp:
	# 	json.dump(new_angles, fp=fp, sort_keys=True, indent=4)

	