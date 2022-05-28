import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import napari
from qtpy.QtCore import QThread, QCoreApplication

if __name__ == "__main__":
	data_path = "/Users/wahab/Data/uCT/"
	ctreader = ctfishpy.CTreader(data_path)
	lump = ctfishpy.Lumpfish()
	master = ctreader.mastersheet()

	# dirty_path = Path("/home/wahab/Data/Local/uCT/low_res")
	dirty_path = Path("/Users/wahab/Data/uCT/51_55")
	# dirty_path = Path("/home/ak18001/Data/HDD/low_res")

	# out_path = Path('/home/ak18001/Data/Local/uCT/low_res_clean')
	out_path = Path('/home/wahab/Data/Local/uCT/low_res_clean')
	temp_angle_path = Path('ctfishpy/Metadata/temp_angles.json')

	fine = [401, 402, 403, 410, 423, 522, 542]
	bad_bois = [424, 429, 433, 434, 435, 465, 467, 468, 534, 543, 559]
	dirty = ['EK_559_566']
	dirty_done = ['EK_423_430', 'EK_431_438', 'EK_464_470', 'EK_533_540', 'EK_541_548', ]
	flipped = [223,262,295,501,543]

	# for n in bad_bois:
	# 	scan, metadata = ctreader.read(n, align=True)
	# 	ctreader.view(scan) 

	original_scale = 100
	detection_scale = 40
	for nums in dirty:
		# path = dirty_path / nums
		path = "/Users/wahab/Data/uCT/MISC/51_55"
		ct, group_metadata = lump.read_dirty(path, r=(500,600), scale=original_scale)
		print(group_metadata)

		# find fish numbers from file name
		start = int(nums.split('_')[-2])
		end = int(nums.split('_')[-1])
		fish_nums = [r for r in range(start, end + 1)]
		print(fish_nums)

		viewer = napari.Viewer(show=False)
		# rescale to save ram
		scale_40 = lump.rescale(ct, detection_scale)
		# detect tubes
		circle_dict = lump.detectTubes(viewer, scale_40)
		# label order
		ordered = lump.labelOrder(viewer, circle_dict)
		# crop
		cropped_cts = lump.crop(ct, ordered, scale=[detection_scale,original_scale])

		for i,cropped in enumerate(cropped_cts):
			num = fish_nums[i]
			if num not in bad_bois:
				continue
			print('num and shape', num, cropped.shape)

			spin_viewer = napari.Viewer(show=False)
			angle, center = lump.spin(spin_viewer, cropped)
			print('angle and center', angle, center)

			metadata = ctreader.read_metadata(num)
			metadata['angle'] = angle
			metadata['center'] = center

			# lump.write_tif(out_path, num, cropped, metadata)

		exit()
