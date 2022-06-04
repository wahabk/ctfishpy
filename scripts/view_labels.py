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

bone = 'Otoliths'
ctreader = ctfishpy.CTreader()
#341,40
for n in [1]:#[527,530,582,589]:
	
	center = ctreader.manual_centers[str(40)]
	roiSize = (128,128,160)

	ct = ctreader.read(n)
	ct = ctreader.crop3d(ct, roiSize, center=center)
	ct = np.array(ct/ct.max(), dtype = np.float32)
	print(ct.max(), ct.min())
	# ct = ctreader.to8bit(ct)
	scan_proj = ctreader.make_max_projections(ct)
	label = ctreader.read_label(bone, n)
	label = ctreader.crop3d(label, roiSize, center=center)
	lab_proj = ctreader.make_max_projections(label)

	projections = ctreader.label_projections(scan_proj, lab_proj)
	print(projections[1].max(), projections[1].min())
	plt.imsave("output/tests/projection.jpg", projections[1])

	# plt.imshow(label[109])
	# plt.show()
	# ctreader.view(ct, label=label)
	
	# ctreader.make_gif(ct, f'output/man_labels{n}.gif', fps=20, label = label)
