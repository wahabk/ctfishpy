import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path


good_auto = [41,43,44,45,46,56,57,79,80,201,203] # these are good segs from 2d unet
wahab_samples 	= [78,200,218,240,277,330,337,341,462,464,364,385]
mariel_samples	= [421,423,242,463,259,459,461]
zac_samples		= [257,443,218,364,464]
# 256 mariel needs to be redone, 
# removing 527, 530, 582, 589
# 421 is barx1
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [40]

organ = 'Otoliths'
ctreader = ctfishpy.CTreader()
segs = 'Otoliths'
#341,40
for n in [527,530,582,589]:
	
	center = ctreader.manual_centers[str(n)]
	roiSize = (160,128,288)

	ct, stack_metadata = ctreader.read(n, align=True)
	ct = ctreader.crop3d(ct, roiSize, center=center)
	label = ctreader.read_label(segs, n, is_amira=True)
	label = ctreader.crop3d(label, roiSize, center=center)

	# plt.imshow(label[109])
	# plt.show()
	# ctreader.view(ct, label=label)
	
	ctreader.make_gif(ct, f'output/man_labels{n}.gif', fps=20, label = label)
