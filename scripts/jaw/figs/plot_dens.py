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
	# dataset_path = '/home/wahab/Data/HDD/uCT/'
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'

	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	data_path = "output/results/jaw/jawunet_data230124.csv"
	cols = ["Dens1","Dens2","Dens3","Dens4","Vol1","Vol2","Vol3","Vol4"]
	sub_bones = ["L_Dentary", "R_Dentary", "L_Quadrate", "R_Quadrate"]
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
	dens = df[cols[:len(sub_bones)]]
	vols = df[cols[len(sub_bones):]]
	dens.columns = sub_bones
	vols.columns = sub_bones

	included = df.index
	included = [i-1 for i in included]
	master = master.iloc[included]

	dens_df = pd.concat([dens, master], axis=1)
	dens_df = ctreader.trim(dens_df, 'genotype', ['wt', 'het', 'hom'])
	dens_df = ctreader.trim(dens_df, 'strain', genes_to_include)
	vols_df = pd.concat([vols, master], axis=1)
	vols_df = ctreader.trim(vols_df, 'genotype', ['wt', 'het', 'hom'])
	vols_df = ctreader.trim(vols_df, 'strain', genes_to_include)

	dens_df.plot()
	plt.savefig("output/results/jaw/dens_df.png")
	plt.clf()

	vols_df.plot()
	plt.savefig("output/results/jaw/vols_df.png")
	plt.clf()

	id_vars = ['age', 'strain', 'genotype', 'length']
	value_vars = sub_bones

	dens_df = dens_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Bone", value_name="Density ($g.cm^{3}HA$)")

	vols_df = vols_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Bone", value_name="Volume ($mm^{3}$)")
	print(dens_df)
	print(vols_df)

	# fig = plt.subplot([331])

	fig, axs = plt.subplots(2,len(sub_bones))
	plt.tight_layout(pad=0.5)

	for i, sub_bone in enumerate(sub_bones):
		print(i, sub_bone)
		dens_sub_bone = ctreader.trim(dens_df, 'Bone', [sub_bone])
		if i == 0:
			ledge = True
		else:
			ledge = False

		# plt.xticks(rotation = -45)
		g = sns.violinplot(data=dens_sub_bone, x='strain', y="Density ($g.cm^{3}HA$)", hue='genotype', ax=axs[0,i], inner='stick',)
		# plt.draw()
		# plt.xticks(rotation=-45) 
		axs[0,i].tick_params(axis='x', rotation=-20)
		axs[0,i].legend().remove()


		vols_sub_bone = ctreader.trim(vols_df, 'Bone', [sub_bone])

		sns.scatterplot(data=vols_sub_bone, x='age', y="Volume ($mm^{3}$)", hue='genotype', ax=axs[1,i], legend=ledge)
	fig.set_figwidth(12)
	fig.set_figheight(5)
	plt.savefig("output/results/jaw/vol_dens_fig.png")
	plt.clf()

	# sns.violinplot(data=dens_df, x='strain', y="Density ($g.cm^{3}HA$)", hue='genotype', inner='stick',)
	# plt.savefig("output/results/jaw/dens_fig.png")
	# plt.clf()

	# sns.violinplot(data=dens_df, x='age', y="Density ($g.cm^{3}HA$)", hue='strain', inner='stick',)
	# plt.savefig("output/results/jaw/dens_age_fig1.png")
	# plt.clf()

	# sns.violinplot(data=dens_df, x='age', y="Density ($g.cm^{3}HA$)", hue='genotype', inner='stick',)
	# plt.savefig("output/results/jaw/dens_age_fig2.png")
	# plt.clf()

	# sns.violinplot(data=vols_df, x='strain', y="Volume ($mm^{3}$)", hue='genotype', inner='stick',)
	# plt.savefig("output/results/jaw/vols_fig.png")
	# plt.clf()

	# sns.violinplot(data=vols_df, x='age', y="Volume ($mm^{3}$)", hue='strain', inner='stick',)
	# plt.savefig("output/results/jaw/vols_age_fig1.png")
	# plt.clf()

	# sns.violinplot(data=vols_df, x='age', y="Volume ($mm^{3}$)", hue='genotype', inner='stick',)
	# plt.savefig("output/results/jaw/vols_age_fig2.png")
	# plt.clf()

	# sns.violinplot(data=vols_df, x='strain', y="Volume ($mm^{3}$)", hue='age', inner='stick',)
	# plt.savefig("output/results/jaw/vols_age_fig3.png")
	# plt.clf()

	# dens_df.plot()
	# plt.savefig("output/results/jaw/dens_df.png")
	# plt.clf()

	# vols_df.plot()
	# plt.savefig("output/results/jaw/vols_df.png")
	# plt.clf()