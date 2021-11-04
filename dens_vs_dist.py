import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy
from copy import deepcopy
import napari
import pandas as pd

if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	centers = ctreader.manual_centers
	organ = 'Otoliths'
	roiSize = (200,200,300)
	nclasses = 3
	mid_wt = [200,330,364,277]
	col_hom = [421,443,582,589]

	data = pd.DataFrame()

	otoliths = ['Lagenal', 'Saccular', 'Utricular']

	for n in [200,582]:
		print(n)
		center = centers[str(n)]

		ct, stack_metadata = ctreader.read_range(n, center, roiSize)
		label = ctreader.read_label(organ, n, is_amira=True)
		label = ctreader.crop3d(label, roiSize, center=center)

		# ctreader.view(ct, label = label)

		for i in range(1, nclasses+1):
			# seperate right and left
			print(i)
			otolith = deepcopy(label)
			otolith[otolith != i] = 0
			otolith[otolith == i] = 1
			otolith, n_labels = scipy.ndimage.label(otolith)
			print(otolith.shape, otolith.max(), n_labels)

			left_right = [1,2]
			for j in left_right:
				center = scipy.ndimage.center_of_mass(ct, labels=otolith, index=j)

				this_otolith = ctreader.crop3d(ct, (50,50,50), center = center)
				
				for i in this_otolith[]

				data[otoliths[i]].append(pd.Series(distances))
				print(center)
				exit()
		
		exit()
		napari.view_labels(otolith)
		napari.run()
		input('read?')
		# exit()
			# ctreader.view(otolith)


			# import pdb; pdb.set_trace()

			# ctreader.view(label[obj], label=label[obj])

			# ctreader.view(label, label=label)


		# find center of label

