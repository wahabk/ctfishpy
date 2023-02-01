import ctfishpy
import napari
import pandas as pd
import numpy as np
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	# import pdb; pdb.set_trace()

	bone = ctreader.JAW

	jaw_curated = [257,351,241,164,50,39,116,441,291,193,420,274,364,401,72,71,69,250,182,183,301,108,216,340,139,337,220,1,154,230,131,133,135,96,98]
	oto_labelled = [200, 240, 256, 259, 330, 341, 385, 421, 443, 461, 463, 527, 582, 78, 218, 242, 257, 277, 337, 364, 40,  423, 459, 462, 464, 530, 589]
	all_nums = master.index.to_list()
	bins = [6,12,24] # for age in months
	conditions = ['wt', 'het', 'hom']
	to_plot = all_nums

	#remove genotypes injected or mosaic etc
	master = master[master['genotype'].isin(conditions)]
	master = master[master.index.isin(to_plot)]
	
	# master = stratified_sample(master, ['age', 'genotype'], size=25, seed = 69)
	# master.to_csv('output/otoliths2seg.csv')
	print(master)
	print(bins)

	# Group all ages into 6, 12, 24, 36 months old
	master.age = pd.cut(master.age, bins = len(bins), labels=bins, right = False)
	ages = [ctreader.trim(master, 'age', [b]) for b in bins]

	# find numbers per bin
	nums = []
	for age in ages:
		num = []
		for c in conditions:
			n = 0
			if str(c) in list(age['genotype']): n = age['genotype'].value_counts()[c]
			num.append(n)
		nums.append(num)
	data = np.array(nums)

	print(data)

	column_names = [None, 'wt', None, 'het', None, 'hom']
	row_names = [[None, b] for b in bins]
	row_names = [i for sublist in row_names for i in sublist] # flatten
	
	fig = plt.figure()
	ax = Axes3D(fig)

	lx = len(data[0])            # Work out matrix dimensions
	ly = len(data[:,0])
	xpos = np.arange(0,lx,1)    # Set up a mesh of positions
	ypos = np.arange(0,ly,1)
	xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

	xpos = xpos.flatten()   # Convert positions to 1D array
	ypos = ypos.flatten()
	zpos = np.zeros(lx*ly)

	dx = 0.5 * np.ones_like(zpos)
	dy = dx.copy()
	dz = data.flatten()

	cs = ['r', 'g', 'b'] * ly
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
	ax.tick_params(axis='both', which='major', labelsize=14)
	ax.w_xaxis.set_ticklabels(column_names)
	ax.w_yaxis.set_ticklabels(row_names)
	ax.set_xlabel('Genotype', fontsize=20, labelpad=20)
	ax.set_ylabel('Age', fontsize=20, labelpad=20)
	ax.set_zlabel('Frequency', fontsize=20, labelpad=20)
	plt.show()