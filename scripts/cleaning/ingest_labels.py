import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
from pathlib2 import Path
import gc


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	# lump = ctfishpy.Lumpfish()
	master = ctreader.master
	
	bone = 'Otoliths'

	wahab_samples 	= [78,200,218,240,277,330,337,341,462,464,364,385]
	mariel_samples	= [421,423,242,463,259,459,461]
	zac_samples		= [257,443]

	all_data = wahab_samples+mariel_samples+zac_samples
	all_data.sort()
	print(all_data, len(all_data))

	keys = ctreader.get_label_keys(bone)
	print(keys, len(keys))
	print([master.iloc[new_n-1]['old_n'] for new_n in keys])
	exit()

	# all_amiras = [200, 240, 256, 259, 330, 341, 385, 421, 443, 461, 463, 527, 582, 78,218, 242, 257, 277, 337, 364, 40,  423, 459, 462, 464, 530, 589,]

	# new_names = [master.loc[master['old_n'] == old_n].index[0] for old_n in all_data]
	# new_names.sort()
	# print(new_names, len(new_names))
	# missing = list(set(keys) - set(new_names))
	# print('missing', missing)

	# exit()
	labels_path = ctreader.dataset_path / 'LABELS' / bone 

	for f in labels_path.iterdir():
		if f.suffix == '.am':
			old_n = int(f.stem)
			print('reading', old_n, f)

			label = ctreader.old_read_label(bone, old_n, is_amira=True)
			new_n = master.loc[master['old_n'] == old_n].index[0]
			print('writing', new_n)
			ctreader.write_label(bone, label, new_n)


	# for new_n in missing:
	# 	print('reading', new_n)
	# 	old_n = master.iloc[new_n-1]['old_n']
	# 	label = ctreader.read_label(bone, old_n, is_amira=True)

	# 	# new_n = master.loc[master['old_n'] == old_n].index[0]

	# 	print('writing', new_n)

	# 	ctreader.write_label(bone, label, new_n)

	