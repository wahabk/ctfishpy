import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import seaborn as sns
import scipy
from sklearn import preprocessing
import random

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
	

	wildtypes = ctreader.trim(master, 'genotype', ['wt'])
	oneyrold_wildtypes = ctreader.trim(wildtypes, 'age', [12])
	oneyrold_wildtypes = list(oneyrold_wildtypes['n'])

	col11s = ctreader.trim(master, 'strain', ['col11a2'])
	col11homs = ctreader.trim(col11s, 'genotype', ['hom'])
	col11homs = list(col11homs['n'])

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
	densities = pd.DataFrame.from_dict(densities, orient='index') #.reset_index() # reset index as n's as new column
	densities.columns=['Lagenal', 'Utricular', 'Saccular']

	genotype = []
	for n in densities.index.tolist():
		n = int(n)
		if n in oneyrold_wildtypes:
			genotype.append('wt')
		elif n in col11homs:
			genotype.append('$col11a2$ -/-')
		else:
			genotype.append(None)
	densities['genotype'] = genotype

	densities = densities.dropna(axis=0)
	densities = densities.melt(['genotype'], var_name='Otoliths', value_name='Density [$g.cm^{3}HA$]')


	# volumes = {key:data[key]['vols'] for key in data}
	# volumes = pd.DataFrame.from_dict(volumes, orient='index').reset_index() # reset index as n's as new column
	# volumes = volumes.dropna(axis=0)
	# volumes.columns=['n', 'Lagenal', 'Utricular', 'Saccular']
	# nums = volumes.n.to_list()
	# volumes['genotype'] = ['$col11a2$ -/-' if int(n) in col11homs else 'wt' for n in nums]
	# master = master.set_index('n')
	# volumes['age'] = [master.loc[int(n)]['age'] for n in nums]
	# volumes = volumes.melt(['n', 'genotype', 'age'], var_name='Otoliths', value_name='Volume [$g.cm^{3}HA$]')
	# volumes = volumes[volumes['Volume [$g.cm^{3}HA$]'] < 0.3]
	# # volumes = volumes.drop()
	# print(volumes)
	# # fopr volumes https://stackoverflow.com/questions/29794959/pandas-add-new-column-to-dataframe-from-dictionary
	
	print(len(oneyrold_wildtypes), len(col11homs))
	#40 and 10

	fig = sns.violinplot(x='Otoliths', y='Density [$g.cm^{3}HA$]', hue='genotype', data=densities, inner='stick', legend=False)
	plt.ylim((0.6,3.0))
	plt.legend(loc='lower right')
	plt.show()

	grouped = densities.groupby(['genotype', 'Otoliths'])
	means = grouped.mean()
	std = grouped.std()

	# import pdb; pdb.set_trace()

	# shapiro wilks for normality
	print(std)
	normality = {}
	significance = {}
	for oto in ['Lagenal', 'Utricular', 'Saccular']:
		wt = densities.loc[(densities['genotype'] == 'wt') & (densities['Otoliths'] == oto), 'Density [$g.cm^{3}HA$]'].tolist()
		mut = densities.loc[(densities['genotype'] == '$col11a2$ -/-') & (densities['Otoliths'] == oto), 'Density [$g.cm^{3}HA$]'].tolist()
		wt = np.array(wt)
		mut = np.array(mut)
		
		if oto == 'Utricular':
			# print(wt)
			# wt = preprocessing.normalize([wt])[0]
			# print(wt)
			# mut = preprocessing.normalize([mut])[0]
			plt.boxplot([wt, mut]); plt.show()
			normality[oto] = [scipy.stats.shapiro(wt), scipy.stats.shapiro(mut)]
			# wt = random.sample(list(wt), 10)
			# mut = random.sample(list(mut), 10)
			significance[oto] = scipy.stats.ttest_ind(wt, mut, equal_var=True)
		else:
			normality[oto] = [scipy.stats.shapiro(wt), scipy.stats.shapiro(mut)]
			plt.boxplot([wt, mut]); plt.show()
			significance[oto] = scipy.stats.mannwhitneyu(wt, mut)
	print(normality)
	print(significance)


	# https://stackoverflow.com/questions/48434391/calculate-t-test-statistic-for-each-group-in-pandas-dataframe

	
	# fig = sns.scatterplot(x='age', y='Volume [$g.cm^{3}HA$]', hue='genotype', data=volumes)
	# plt.show()
	

