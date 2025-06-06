import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy
from copy import deepcopy
import napari
import pandas as pd
import math
import seaborn as sns

sns.set(font_scale=1.07)

if __name__ == '__main__':
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"
	
	ctreader = ctfishpy.CTreader(dataset_path)
	centers = ctreader.otolith_centers
	organ = ctreader.OTOLITHS
	roiSize = (200,200,300)
	nclasses = 3
	mid_wt = [200,330,364,277]
	col_hom = [421,443,582,589]
	
	# keys = ctreader.get_hdf5_keys("/home/wahab/Data/HDD/uCT/LABELS/OTOLITHS/OTOLITHS.h5")
	# print(keys)

	data = []

	otolith_names = ['background', 'Lagenal', 'Saccular', 'Utricular']

	for n, old_n in zip([64,420], [200, 582]) :
		center = centers[n]

		# ct, stack_metadata = ctreader.read_range(n, center, roiSize)
		ct = ctreader.read(n)
		ct = ctreader.crop3d(ct, roiSize, center=center)
		stack_metadata = ctreader.read_metadata(n)
		label = ctreader.read_label(organ, n, is_amira=False)
		# label = ctreader.read_label(organ, old_n, is_amira=True)
		# if old_n == 200: label  = ctreader.rotate_array(label, 94, is_label=True)
		label = ctreader.crop3d(label, roiSize, center=center)

		# ctreader.view(ct, label = label)

		for i in range(1, nclasses+1):
			# seperate right and left
			otolith = deepcopy(label)
			otolith[otolith != i] = 0
			otolith[otolith == i] = 1
			otolith, n_labels = scipy.ndimage.label(otolith) # label left and right

			left_right = [1,2]
			for j in left_right:
				com = scipy.ndimage.center_of_mass(ct, labels=otolith, index=j)
				this_otolith = ctreader.crop3d(ct, (50,50,50), center = com)
				this_label = ctreader.crop3d(otolith, (50,50,50), center = com)
				

				genotype = 'wt' if old_n in mid_wt else 'col11a2'
				name = otolith_names[i]
				print(n,i,j,name,genotype,com)

				# napari.view_labels(this_label)
				# napari.run()
				# import pdb; pdb.set_trace()

				indices = np.where(this_label==j)
				com = scipy.ndimage.center_of_mass(this_otolith, labels=this_label, index=j)

				for index in zip(indices[0], indices[1], indices[2]):
					density = this_otolith[index]*0.0000381475547417411
					distance = math.dist(index, com)
					distance = float(distance)* float(stack_metadata['VoxelSizeX']) * 100
					data.append([n,genotype,name,distance,density])

	df = pd.DataFrame(data, columns=['n', 'Genotype', 'Otolith', 'Distance [$\mu$m]', 'Density [$g.cm^{3}HA$]'])

	grouped = df.groupby(['n', 'Genotype', 'Otolith'])
	print(grouped.mean())


	fig, axes = plt.subplots(1,3, figsize=(15,4))
	for ax, oto_name in zip(axes, otolith_names[1:]):
		this_df = df.where(df.Otolith == oto_name).dropna()
		sns.histplot(this_df, x='Distance [$\mu$m]', y='Density [$g.cm^{3}HA$]', hue='Genotype', bins=30, ax=ax, cbar=True)
		# ax.
	plt.tight_layout(pad=0.05)




	plt.show()


	
	exit()
	napari.view_labels(otolith)
	napari.run()
	input('read?')

