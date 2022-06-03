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
	
	organ = 'Otoliths'

	wahab_samples 	= [78,200,218,240,277,330,337,341,462,464,364,385]
	mariel_samples	= [421,423,242,463,259,459,461]
	zac_samples		= [257,443,218,364,464]
	labels_path = ctreader.dataset_path / 'LABELS' / organ 

	all_data = wahab_samples+mariel_samples+zac_samples

	all_amiras_27 = ['364', '240', '461', '459', '337', '385', '464', '421', '200', '256', '257', '443', '40', '277', '530', '341', '527', '582', '330', '218', '462', '259', '242', '423', '463', '78', '589']

	for f in labels_path.iterdir():
		if f.suffix == '.am':
			old_n = int(f.stem)
			print('reading', old_n, f)

			label = ctreader.read_label(organ, old_n, is_amira=True)
			new_n = master.loc[master['old_n'] == old_n].index[0]
			print('writing', new_n)
			ctreader.write_label(organ, label, new_n)

	# for old_n in all_data:
	# 	print('reading', old_n)
	# 	label = ctreader.read_label(organ, old_n, is_amira=True)

	# 	new_n = master.loc[master['old_n'] == old_n].index[0]

	# 	print('writing', new_n)

	# 	ctreader.write_label(organ, label, new_n)