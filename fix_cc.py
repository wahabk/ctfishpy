import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json



if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	sample = [76, 40, 81, 85, 88, 222, 236, 218, 425]
	centres = {}
	for fish in sample:
		positions = ctreader.cc_fixer(fish)
		centres[str(fish)] = positions
	with open('cc_centres.json', 'w') as f:
		json.dump(centres, f, sort_keys=True, indent=4)
	with open('cc_centres.json', 'r') as fp:
		data = json.load(fp)
	print(data)
		