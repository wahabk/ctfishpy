import ctfishpy
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	ctreader = ctfishpy.CTreader(data_path=dataset_path)
	master = ctreader.mastersheet()

	data_path = "output/results/2d_unet_results.csv"

	# with open(data_path, 'r') as fr:
	# 	data = json.load(fr)
	# new_data = {}
	# for k in data.keys():
	# 	new_data[k] = {
	# 		'dens1': data[k]['dens'][0],
	# 		'dens2': data[k]['dens'][1],
	# 		'dens3': data[k]['dens'][2],
	# 		'vols1': data[k]['vols'][0],
	# 		'vols2': data[k]['vols'][1],
	# 		'vols3': data[k]['vols'][2],
	# 	}
	# df = pd.DataFrame.from_dict(new_data, orient='index')

	cols = ['Density1', 'Density2', 'Density3', 'Volume1', 'Volume2', 'Volume3']
	otos = ["Lagenal", "Saccular", "Utricular"]

	df = pd.read_csv(data_path, index_col=0)
	dens = df[cols[:3]]
	vols = df[cols[3:]]
	print(df)

	dens.columns = otos
	vols.columns = otos

	included = df.index
	included = [i-1 for i in included]
	metadata = master.iloc[included]

	dens_df = pd.concat([dens, metadata], axis=1)
	vols_df = pd.concat([vols, metadata], axis=1)

	['Lagenal', 'Saccular', 'Utricular', 'ak_n', 'Dataset', 'old_n', 'age',
       'age(old)', 'genotype', 'strain', 'name', 'shape', 'size', 'VoxelSizeX',
       'VoxelSizeY', 'VoxelSizeZ', 're-uCT scan', 'Comments', 'Phantom',
       'Scaling Value', 'Arb Value', 'angle', 'center']
	
	id_vars = ['ak_n', 'age', 'strain', 'genotype']
	value_vars = otos

	dens_df = dens_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Otolith", value_name="Density")

	vols_df = vols_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Otolith", value_name="Volume")

	# df.dropna(subset = ["Density1",  "Density2",  "Density3",   "Volume1",   "Volume2",   "Volume3"])
	# Make big dataframe with ages and genotypes and vols and dens

	import pdb; pdb.set_trace()

	sns.violinplot(data=dens_df, x='genotype', y="Density", hue='Otolith')
	plt.show()

	sns.scatterplot(data=vols_df, x='age', y="Volume", hue='Otolith')
	plt.show()

