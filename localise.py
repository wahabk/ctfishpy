import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	datapath = ctreader.centres_path
	fishnums = ctreader.fish_nums

	# wahab_samples 	= [200,218,240,277,330,337,341,462,40,78, 464,385] # cc doesnt work on 464 385
	# mariel_samples	= [242,256,259,459,463,530,589,421,423] # 421 423 removed
	# zac_samples		= [257,443,461,527,582]

	with open(datapath, 'r') as fp:
		centers = json.load(fp)
	done = list(centers.keys())

	for fish in [464, 385]:
		# if str(fish) not in done:
			print(fish)
			center = ctreader.cc_fixer(fish)
			centers[str(fish)] = center
	# with open(datapath, 'w') as f:
	# 	json.dump(centers, f, sort_keys=True, indent=4)
	# with open(datapath, 'r') as fp:
	# 	centers = json.load(fp)
	print(centers)
