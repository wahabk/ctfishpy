import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json



if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	datapath = ctreader.centres_path

	wahab_samples 	= [200,218,240,277,330,337,341,462,40,78, 464,385] # cc doesnt work on 464 385
	mariel_samples	= [242,256,259,459,463,530,589,421,423] # 421 423 removed
	zac_samples		= [257,443,461,527,582]

	sample = wahab_samples+mariel_samples+zac_samples

	with open(datapath, 'r') as fp:
		data = json.load(fp)
	done = list(data.keys())

	for fish in sample:
		if str(fish) not in done:
			print(fish)
			positions = ctreader.cc_fixer(fish)
			data[str(fish)] = positions
	with open(datapath, 'w') as f:
		json.dump(data, f, sort_keys=True, indent=4)
	with open(datapath, 'r') as fp:
		data = json.load(fp)
	print(data)
		