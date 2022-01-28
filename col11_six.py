import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json, csv
from pathlib2 import Path
import seaborn as sns
import scipy
from sklearn import preprocessing
import random

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	master = ctreader.mastersheet()
	datapath = 'output/otolith_data.csv'
	datapath_col11 = 'output/col11_new_data.json'

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
	sixmonth_wildtypes = ctreader.trim(wildtypes, 'age', [6,7,])
	sixmonth_wildtypes = list(sixmonth_wildtypes['n'])

	# col11s = ctreader.trim(master, 'strain', ['col11a2'])
	# col11homs = ctreader.trim(col11s, 'genotype', ['hom'])
	# col11homs = list(col11homs['n'])

	with open(datapath, 'r') as fr:
		data = json.load(fr)
	with open(datapath_col11, 'r') as fr:
		data_col11 = json.load(fr)

	dens_wt = {key:data[key]['densities'] for key in data}
	dens_col11 = {key:data_col11[key]['densities'] for key in data_col11}


	densities = {**dens_col11, **dens_wt}

	densities = pd.DataFrame.from_dict(densities, orient='index') #.reset_index() # reset index as n's as new column
	densities.columns=['Lagenal', 'Saccular', 'Utricular']

	genotype = []
	for n in densities.index.tolist():
		n = int(n)
		if n in sixmonth_wildtypes:
			genotype.append('wt 6(m)')
		elif str(n) in dens_col11.keys():
			genotype.append('$col11a2$ -/- 6(m)')
		else:
			genotype.append(None)
	densities['genotype'] = genotype

	densities = densities.dropna(axis=0)
	densities = densities.melt(['genotype'], var_name='Otoliths', value_name='Density ($g.cm^{3}HA$)')

	densities.to_csv('output/yushipaper/six_month.csv')

	
	print(len(sixmonth_wildtypes), len(data_col11))
	#40 and 10

	fig = sns.violinplot(x='Otoliths', y='Density ($g.cm^{3}HA$)', hue='genotype', data=densities, inner='stick', legend=False)
	plt.ylim((0.5,3.0))
	plt.legend(loc='lower right')
	plt.savefig('output/yushipaper/sixMonthDens.png')
	
	grouped = densities.groupby(['genotype', 'Otoliths'])
	means = grouped.mean()
	std = grouped.std()


	# shapiro wilks for normality
	print(std)
	normality = {}
	significance = {}
	for oto in ['Lagenal', 'Saccular', 'Utricular']:
		wt = densities.loc[(densities['genotype'] == 'wt 6(m)') & (densities['Otoliths'] == oto), 'Density ($g.cm^{3}HA$)'].tolist()
		mut = densities.loc[(densities['genotype'] == '$col11a2$ -/- 6(m)') & (densities['Otoliths'] == oto), 'Density ($g.cm^{3}HA$)'].tolist()
		wt = np.array(wt)
		mut = np.array(mut)
		
		if oto == 'Utricular':
			# print(wt)
			# wt = preprocessing.normalize([wt])[0]
			# print(wt)
			# mut = preprocessing.normalize([mut])[0]
			# plt.boxplot([wt, mut]); plt.show()
			normality[oto] = [scipy.stats.shapiro(wt), scipy.stats.shapiro(mut)]
			# wt = random.sample(list(wt), 10)
			# mut = random.sample(list(mut), 10)
			significance[oto] = scipy.stats.ttest_ind(wt, mut, equal_var=True)
		else:
			normality[oto] = [scipy.stats.shapiro(wt), scipy.stats.shapiro(mut)]
			# plt.boxplot([wt, mut]); plt.show()
			significance[oto] = scipy.stats.mannwhitneyu(wt, mut)
	print(normality)
	print(significance)

	rt_path = 'output/yushipaper/col11a2-relax-time.csv'
	rt = pd.read_csv(rt_path).to_dict()

	densities = pd.DataFrame.from_dict(dens_col11, orient='index').to_dict()

	# import pdb; pdb.set_trace()

	data = [{'Relaxation time (s)': rt['relax_time(s)'][int(k)-1], 
			'Lagenal': densities[0][k], 
			'Saccular': densities[1][k], 
			'Utricular': densities[2][k]}  
			for k in densities[0].keys()]
	df = pd.DataFrame.from_dict(data)

	print(df)
	plt.clf()
	plt.cla()

	df = df.melt(['Relaxation time (s)'], var_name='Otoliths', value_name='Density ($g.cm^{3}HA$)')
	
	print(df)

	sns.lineplot(x='Relaxation time (s)', y='Density ($g.cm^{3}HA$)', hue='Otoliths' , data=df)
	plt.savefig('output/yushipaper/rt_v_dens.png')


	# https://stackoverflow.com/questions/48434391/calculate-t-test-statistic-for-each-group-in-pandas-dataframe	
	# fig = sns.scatterplot(x='age', y='Volume [$g.cm^{3}HA$]', hue='genotype', data=volumes)
	# plt.show()



	

