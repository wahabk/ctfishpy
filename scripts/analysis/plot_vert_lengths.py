import ctfishpy

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch
import seaborn as sns
import scipy



if __name__ == '__main__':
	dataset_path = '/home/wahab/Data/HDD/uCT/'
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	# dataset_path = '/mnt/scratch/ak18001/uCT/'

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master


	data_path = 'output/results/vert_lengths2.csv'
	df = pd.read_csv(data_path, index_col=0)

	wt = [36,37,38,39,40,41,42,225,226,229,]
	col11 = [456,457,458,459,460,461,462,463,464,465,]

	df['Genotype'] = ['wt']*10 + ['$\it{col11a2 -/-}$']*10

	print(df)

	sns.boxplot(data=df, x="Genotype", y = "Length (mm)", palette="Dark2")
	plt.show()

	wt = ctreader.trim(df, 'Genotype', ['wt'])['Length (mm)'].to_numpy()
	mut = ctreader.trim(df, 'Genotype', ['$\it{col11a2 -/-}$'])['Length (mm)'].to_numpy()

	# wt = np.array(wt)
	# mut = np.array(mut)
	normality = [scipy.stats.shapiro(wt), scipy.stats.shapiro(mut)]

	print(normality)

	significance = scipy.stats.ttest_ind(wt, mut, equal_var=False)

	print(significance)