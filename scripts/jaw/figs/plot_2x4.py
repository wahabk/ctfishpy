import ctfishpy

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch
import seaborn as sns


def get_names(df, col='genotype'):
	order =['wt', 'het', 'hom', 'mosaic']
	names = ""
	df_dict = df.groupby(col)[col].count().to_dict()
	
	for i, o in enumerate(order):
		
		try: 
	  		names = names + f"{df_dict[o]}, " 
		except: pass
			
	
	return names

if __name__ == '__main__':
	# dataset_path = '/home/wahab/Data/HDD/uCT/'
	# dataset_path = '/home/ak18001/Data/HDD/uCT/'
	dataset_path = '/mnt/scratch/ak18001/uCT/'
	sns.color_palette("hls", 3)


	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master
	print(master)

	data_path = "output/results/jaw/jawunet_data230124.csv"
	cols = ["Dens1","Dens2","Dens3","Dens4","Vol1","Vol2","Vol3","Vol4"]
	sub_bones = ["L_Dentary", "R_Dentary", "L_Quadrate", "R_Quadrate"]
	all_genes = ['wt', 'barx1', 'arid1b', 'col11a2', 'panther', 'giantin', 'chst11', 'runx2',
	'wnt16', 'ncoa3', 'gdf5', 'mcf2l', 'dot1', 'scxb', 'scxa', 'col9', 'sp7', 'col11 ',
	'vhl', 'tert', 'chsy1', 'col9a1', 'ras', 'atg', 'ctsk', 'spp1', 'il10', 'il1b',
	'irf6',]
	genes_to_include = ['wt', 'barx1', 'arid1b', 'col11a2', 'giantin', 'chst11', 'runx2',
	'wnt16', 'ncoa3', 'gdf5', 'mcf2l', 'scxa', 'sp7', 'col11 ',
	'chsy1', 'atg', 'ctsk', 'spp1']
	genes_to_analyse = ['wt','col11 ']
	age_bins = [6,12] # for age in months

	# import pdb; pdb.set_trace()

	# bin  ages
	master.age = pd.cut(master.age, bins = len(age_bins), labels=age_bins, 
				include_lowest=True)
	# ages = [ctreader.trim(master, 'age', [b]) for b in bins]
	print(master)
	print(master.strain.to_list())
	print(master.genotype.to_list())

	# clean strain naming
	for n in master.index:
		strain = master['strain'].loc[n]
		geno = master['genotype'].loc[n]
		# print(strain, geno)
		if geno == 'wt': 
			master['strain'].loc[n] = 'wt'

	df = pd.read_csv(data_path, index_col=0)
	dens = df[cols[:len(sub_bones)]]
	vols = df[cols[len(sub_bones):]]
	dens.columns = sub_bones
	vols.columns = sub_bones

	# included = df.index
	# included = [i-1 for i in included]
	# master = master.iloc[included]

	dens_df = pd.concat([dens, master], axis=1)
	# dens_df = ctreader.trim(dens_df, 'genotype', ['wt', 'het', 'hom'])
	dens_df = ctreader.trim(dens_df, 'strain', genes_to_analyse)
	vols_df = pd.concat([vols, master], axis=1)
	# vols_df = ctreader.trim(vols_df, 'genotype', ['wt', 'het', 'hom'])
	vols_df = ctreader.trim(vols_df, 'strain', genes_to_analyse)

	print(dens_df.strain.to_list())

	id_vars = ['age', 'strain', 'genotype', 'length']
	value_vars = sub_bones

	# melt dataframes
	dens_df = dens_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Bone", value_name="Density ($g.cm^{3}HA$)")
	vols_df = vols_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Bone", value_name="Volume ($mm^{3}$)")
	print(dens_df)
	print(vols_df)

	fig , axs = plt.subplots(len(sub_bones)//2, len(age_bins),sharey="row",sharex="col")
	plt.tight_layout()

	bones_to_analyse = ["Dentary","Quadrate"]
	bones_to_analyse_short = ["Dent","Quad"]
	fig_index=0
	for i, b in enumerate(bones_to_analyse):
		df = ctreader.trim(dens_df, "Bone", [f"L_{b}", f"R_{b}"])
		for j, bin_ in enumerate(age_bins):
			print(i,j,b,bin_)

			final_df = df.loc[df['age'] == bin_]
			final_df.to_csv("output/test.csv")
			# dropna?
			if len(df)>0:
				# sns.violinplot(data=dens_df, x="genotype", y="Density ($g.cm^{3}HA$)", ax=axs[i,j], inner='stick',)
				sns.boxplot(data=final_df,  x="genotype", y="Density ($g.cm^{3}HA$)",ax=axs[i,j])
			else: print("\nSKIPPED\n")
			if b == "Dentary": axs[i,j].set_xlabel("")
			if bin_ == 12: axs[i,j].set_ylabel("")
			names = get_names(final_df)
			sub_plot_title = f"{chr(fig_index+65)} - {bones_to_analyse_short[i]} {bin_} months, n = {names}"
			axs[i,j].set_title(sub_plot_title)

			fig_index += 1

	# import pdb;pdb.set_trace()
	
	fig.set_figwidth(7)
	fig.set_figheight(9)
	fig.suptitle(f"Densities {genes_to_analyse[-1]}", fontsize=16, y=1.025)

	# plt.subplots_adjust(left=0.125, bottom=0.125, right=0.25, top=0.25, wspace=0.25, hspace=0.25)
	# plt.gcf().subplots_adjust(bottom=0.5)
	plt.savefig(f"output/results/jaw/genes/dens_{genes_to_analyse[-1]}.png", bbox_inches="tight")
	plt.clf()

	fig , axs = plt.subplots(len(sub_bones)//2, len(age_bins),sharey="row",sharex="col")
	plt.tight_layout()
	fig_index = 0
	for i, b in enumerate(bones_to_analyse):
		df = ctreader.trim(vols_df, "Bone",[f"L_{b}", f"R_{b}"])
		for j, bin_ in enumerate(age_bins):
			print(b,bin_)
			print(df)

			final_df = df.loc[df['age'] == bin_]

			if len(df)>0:
				# sns.violinplot(data=dens_df, x="genotype", y="Density ($g.cm^{3}HA$)", ax=axs[i,j], inner='stick',)
				sns.boxplot(data=final_df, x="genotype", y="Volume ($mm^{3}$)", ax=axs[i,j])
			else: print("\nSKIPPED\n")
   
			if b == "Dentary": axs[i,j].set_xlabel("")
			if bin_ == 12: axs[i,j].set_ylabel("")
			names = get_names(final_df)
			sub_plot_title = f"{chr(fig_index+65)} - {bones_to_analyse_short[i]} {bin_} months, n = {names}"
			axs[i,j].set_title(sub_plot_title)

			fig_index += 1

	fig.set_figwidth(7)
	fig.set_figheight(9)
	# plt.subplots_adjust(left=0.125, bottom=0.125, right=0.25, top=0.25, wspace=0.25, hspace=0.25)
	# plt.gcf().subplots_adjust(bottom=0.05)
	fig.suptitle(f"Volumes {genes_to_analyse[-1]}", fontsize=16, y=1.025)
	plt.savefig(f"output/results/jaw/genes/vols_{genes_to_analyse[-1]}.png", bbox_inches="tight")





