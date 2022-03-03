import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
from pathlib2 import Path

if __name__ == "__main__":
	dataset_path = '/home/wahab/Data/HDD/uCT/'
	ctreader = ctfishpy.CTreader()
	datapath = ctreader.centres_path
	fishnums = ctreader.fish_nums

	# wahab_samples 	= [200,218,240,277,330,337,341,462,40,78, 464,385] # cc doesnt work on 464 385
	# mariel_samples	= [242,256,259,459,463,530,589,421,423] # 421 423 removed
	# zac_samples		= [257,443,461,527,582]


	dataset_path = Path('/home/ak18001/Data/HDD/uCT/Misc/yushi_data/n')

	with open(datapath, 'r') as fp:
		centers = json.load(fp)
	done = list(centers.keys())

	centers = {}
	for fish in dataset_path.iterdir():
		# if str(fish) not in done:
		if fish.is_dir():
			print(fish)

			ct = ctreader.read_path(fish)
			# ctreader.view(ct)

			# import pdb; pdb.set_trace()

			projections = ctreader.make_max_projections(ct)
			projections = [ctreader.to8bit(p) for p in projections]

			center = ctreader.cc_fixer(projections)
			print(center)
			centers[str(fish.stem)] = center


			# center = centers[str(fish)]
			# ct, metadata = ctreader.read(fish, align=True)
			# otoliths = ctreader.crop3d(ct, (200,200,200), center=center)
			# ctreader.view(otoliths)


	jsonpath = dataset_path / 'centers.json'
	with open(jsonpath, 'w') as f:
		json.dump(centers, f, sort_keys=True, indent=4)
	with open(jsonpath, 'r') as fp:
		centers = json.load(fp)
	print(centers)
