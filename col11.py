import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import seaborn as sns

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	master = ctreader.mastersheet()
	datapath = 'output/otolith_data.csv'

	# strains = ['col11a2']
	# master = master[master['strain'].isin(strains)]
	# genotypes = ['wt']
	# master = master[master['genotype'].isin(genotypes)]
	# ages = ['12']
	# master = master[master['age'].isin(ages)]
	# print(master)
	# print(list(master['n']))
	# print(list(master['age']))

	col11s = ctreader.trim(master, 'strain', ['col11a2'])
	col11homs = list(ctreader.trim(col11s, 'genotype', ['hom'])['n'])


	homs = [421, 443, 582, 583, 584, 585, 586, 587, 588, 589]
	het7month = [459, 460, 461]
	het18month = [256, 257, 258, 259]
	het28month = [462, 463, 464]
	wt12month = [68, 69, 70, 71, 72, 73, 74, 102, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 247, 431, 432, 433, 434, 440, 441, 442, 574, 575, 576]

	# for n in homs+het18month+het28month+het7month:
	# 	ct, metadata = ctreader.read(n, align=True)
	# 	center = ctreader.manual_centers[str(n)]

	# 	otolith = ctreader.crop3d(ct, [200,200,200], center=center)
	# 	ctreader.view(otolith)

	with open(datapath, 'r') as fr:
		data = json.load(fr)


	densities = {key:data[key]['densities'] for key in data}
	densities = pd.DataFrame.from_dict(densities, orient='index').reset_index() # reset index as n's as new column
	densities = densities.dropna(axis=0)
	densities.columns=['n', 'Lagenal', 'Utricular', 'Saccular']
	nums = densities['n']
	densities['genotype'] = ['col11' if int(n) in col11homs else 'wt' for n in nums]
	densities = densities.melt(['n', 'genotype'], var_name='Otoliths', value_name='Density [units]')


	volumes = {key:data[key]['vols'] for key in data}
	volumes = pd.DataFrame.from_dict(volumes, orient='index').reset_index() # reset index as n's as new column
	volumes = volumes.dropna(axis=0)
	volumes.columns=['n', 'Lagenal', 'Utricular', 'Saccular']
	nums = volumes.n.to_list()
	volumes['genotype'] = ['col11' if int(n) in col11homs else 'wt' for n in nums]
	master = master.set_index('n')
	volumes['age'] = [master.loc[int(n)]['age'] for n in nums]
	volumes = volumes.melt(['n', 'genotype', 'age'], var_name='Otoliths', value_name='Volume [units]')
	volumes = volumes[volumes['Volume [units]'] < 0.3]
	# volumes = volumes.drop()
	print(volumes)
	# fopr volumes https://stackoverflow.com/questions/29794959/pandas-add-new-column-to-dataframe-from-dictionary
	
	fig = sns.violinplot(x='Otoliths', y='Density [units]', hue='genotype', data=densities)
	plt.show()

	fig = sns.scatterplot(x='age', y='Volume [units]', hue='genotype', data=volumes)
	plt.show()
	

