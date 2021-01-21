import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json



if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	datapath = ctreader.dataset_path / 'cc_centres_otoliths.json'

	sample = [40,78,200,218,240,277,330,337,341,462,464,364]
	centres = {}
	for fish in sample:
		positions = ctreader.cc_fixer(fish)
		centres[str(fish)] = positions
	with open(datapath, 'w') as f:
		json.dump(centres, f, sort_keys=True, indent=4)
	with open(datapath, 'r') as fp:
		data = json.load(fp)
	print(data)
		