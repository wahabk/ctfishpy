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
auto = [41,43,44,45,46,56,57,70,78,79,80,90,92,200,201,203] # these are good segs from 2d unet

segs = 'Otoliths_unet2d'

for n in auto:
	
	center = ctreader.manual_centers[str(n)]
	roiSize = (160,128,288,)

	ct, stack_metadata = ctreader.read(n, align=True)
	ct = ctreader.crop3d(ct, roiSize, center=center)
	label = ctreader.read_label(segs, n, is_amira=False)
	label = ctreader.crop3d(label, roiSize, center=center)

	ctreader.view(ct, label=label)
	
	# # ctreader.make_gif(ct[1200:1500], 'output/test_labels.gif', fps=20, label = label[1200:1500])
