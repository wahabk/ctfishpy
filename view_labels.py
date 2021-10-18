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

segs = 'Otoliths_unet2d'

for n in [464, 40, 582]: #[40, 421,582, 461, 464, 583, 584, 585, 586, 587]:#ctreader.fish_nums[ctreader.fish_nums.index(87)+1:]:
	
	center = ctreader.manual_centers[str(n)]
	roiSize = (192,288,128)

	ct, stack_metadata = ctreader.read(n, align=True)
	ct = ctreader.crop3d(ct, (192,288,128), center=center)
	label = ctreader.read_label(segs, n, is_amira=False)
	label = ctreader.crop3d(label, (192,288,128), center=center)

	ctreader.view(ct, label=label)
	
	# # ctreader.make_gif(ct[1200:1500], 'output/test_labels.gif', fps=20, label = label[1200:1500])
