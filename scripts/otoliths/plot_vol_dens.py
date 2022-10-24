import ctfishpy

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch
import seaborn as sns



if __name__ == '__main__':
	dataset_path = '/home/wahab/Data/HDD/uCT/'
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	# dataset_path = '/mnt/scratch/ak18001/uCT/'

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	data_path = "output/results/3d_unet_data20221024.csv"

	cols = ['Dens1', 'Dens2', 'Dens3', 'Vol1', 'Vol2', 'Vol3']
	otos = ["Lagenal", "Saccular", "Utricular"]
	all_genes = ['wt', 'barx1', 'arid1b', 'col11a2', 'panther', 'giantin', 'chst11', 'runx2',
	'wnt16', 'ncoa3', 'gdf5', 'mcf2l', 'dot1', 'scxb', 'scxa', 'col9', 'sp7', 'col11 ',
	'vhl', 'tert', 'chsy1', 'col9a1', 'ras', 'atg', 'ctsk', 'spp1', 'il10', 'il1b',
	'irf6', 'col11a3', 'col11a4', 'col11a5', 'col11a6', 'col11a7', 'col11a8',
	'col11a9', 'col11a10', 'col11a11']
	genes_to_include = ['wt', 'barx1', 'arid1b', 'col11a2', 'giantin', 'chst11', 'runx2',
	'wnt16', 'ncoa3', 'gdf5', 'mcf2l', 'scxa', 'sp7', 'col11 ',
	'chsy1', 'atg', 'ctsk', 'spp1']

	# import pdb; pdb.set_trace()
	#clean strain naming
	for n in master.index:
		strain = master['strain'].loc[n]
		geno = master['genotype'].loc[n]
		# print(strain, geno)
		if geno == 'wt': 
			master['strain'].loc[n] = 'wt'

	print(master['strain'].unique())

	df = pd.read_csv(data_path, index_col=0)
	print(df)
	dens = df[cols[:3]]
	vols = df[cols[3:]]
	dens.columns = otos
	vols.columns = otos

	included = df.index
	included = [i-1 for i in included]
	master = master.iloc[included]

	dens_df = pd.concat([dens, master], axis=1)
	dens_df = ctreader.trim(dens_df, 'genotype', ['wt', 'het', 'hom'])
	dens_df = ctreader.trim(dens_df, 'strain', genes_to_include)
	vols_df = pd.concat([vols, master], axis=1)
	vols_df = ctreader.trim(vols_df, 'genotype', ['wt', 'het', 'hom'])
	vols_df = ctreader.trim(vols_df, 'strain', genes_to_include)

	id_vars = ['age', 'strain', 'genotype', 'length']
	value_vars = otos

	dens_df = dens_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Otolith", value_name="Density")

	vols_df = vols_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Otolith", value_name="Volume")
	print(dens_df)
	print(vols_df)

	# fig = plt.subplot([331])

	fig, axs = plt.subplots(2,3)

	for i, oto in enumerate(otos):
		dens_oto = ctreader.trim(dens_df, 'Otolith', [oto])
		if i == 0:
			l = True
		else:
			l = False

		# plt.xticks(rotation = -45)
		g = sns.boxplot(data=dens_oto, x='strain', y="Density", hue='genotype', ax=axs[0,i])
		# plt.draw()
		# plt.xticks(rotation=-45) 
		axs[0,i].tick_params(axis='x', rotation=-45)


		vols_oto = ctreader.trim(vols_df, 'Otolith', [oto])

		sns.scatterplot(data=vols_oto, x='age', y="Volume", hue='genotype', ax=axs[1,i], legend=l)
	plt.show()


	exit()

	sns.violinplot(data=dens_df, x='strain', y="Density", hue='genotype')
	plt.show()

