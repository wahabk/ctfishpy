import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path



mariele_samples 	= [40,242,256,259,421,423,459,463,530,589]
zac_samples 		= [443,461,527,582]
my_samples		= [78,200,218,240,330,337,341,364,385,462,464]
organ = 'Otoliths'
ctreader = ctfishpy.CTreader()

my_samples 	= [78,200,240,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,218,464,364,385]
check = mariele_samples+zac_samples+my_samples

for n in check:
	ct, stack_metadata = ctreader.read(n, align=True)
	datapath = ctreader.dataset_path
	align = True if n in [78,200,218,240,277,330,337,341,462,464,364,385] else False
	label = ctreader.read_label(organ, n, align=align, is_amira=True)
	center = ctreader.manual_centers[str(n)]

	roiSize = (192,288)

	ct = ctreader.crop_around_center3d(ct, roiSize=roiSize, center=center, roiZ=128)
	label = ctreader.crop_around_center3d(label, roiSize=roiSize, center=center, roiZ=128)
	
	ctreader.view(ct, label=label)
	# ctreader.make_gif(ct[1200:1500], 'output/test_labels.gif', fps=20, label = label[1200:1500])
