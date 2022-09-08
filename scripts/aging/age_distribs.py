from cmath import nan
from doctest import master
import ctfishpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib2 import Path
import napari



if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/uCT'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dataset_path = '/home/wahab/Data/HDD/uCT'


	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	trimmed = ctreader.trim(master, 'genotype', ['het', 'hom', 'wt'])
	droppedgeno = master[~master.genotype.isin(['het', 'hom', 'wt'])]
	df = trimmed[trimmed['age'].notna()]
	dropped_age = trimmed[trimmed['age'].isna()]
	print(len(dropped_age))
	print(len(droppedgeno))

	ages = df['age'].values
	print(type(ages))
	# print(ages)
	print(ages.dtype)
	# print(np.argwhere(np.isnan(ages)))

	print(len(ages), ages.max(), ages.min(), ages.mean(), ages.std())

	

	sns.boxplot(data=master, x='age')
	# plt.scatter(ages)
	plt.show()

	# TODO use this to fit a least squares
	# https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html

	sns.scatterplot(data=df, x='age', y='length', hue='genotype')
	plt.show()

	# TODO find volumes of whole fish


