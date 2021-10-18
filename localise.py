import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader('/home/wahab/Data/HDD/uCT/')
	datapath = ctreader.centres_path
	fishnums = ctreader.fish_nums

	# wahab_samples 	= [200,218,240,277,330,337,341,462,40,78, 464,385] # cc doesnt work on 464 385
	# mariel_samples	= [242,256,259,459,463,530,589,421,423] # 421 423 removed
	# zac_samples		= [257,443,461,527,582]

	with open(datapath, 'r') as fp:
		centers = json.load(fp)
	done = list(centers.keys())

	# centers = {}

	for fish in [421]:
		# if str(fish) not in done:
		print(fish)

		center = ctreader.cc_fixer(fish)
		print(center)
		centers[str(fish)] = center


		# center = centers[str(fish)]
		# ct, metadata = ctreader.read(fish, align=True)
		# otoliths = ctreader.crop3d(ct, (200,200,200), center=center)
		# ctreader.view(otoliths)



	with open(datapath, 'w') as f:
		json.dump(centers, f, sort_keys=True, indent=4)
	with open(datapath, 'r') as fp:
		centers = json.load(fp)
	print(centers)
