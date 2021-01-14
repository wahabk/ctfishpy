import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# the functions:
def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
	'''
	It samples data from a pandas dataframe using strata. These functions use
	proportionate stratification:
	n1 = (N1/N) * n
	where:
		- n1 is the sample size of stratum 1
		- N1 is the population size of stratum 1
		- N is the total population size
		- n is the sampling size
	
	https://www.kaggle.com/flaviobossolan/stratified-sampling-python
	
	Parameters
	----------
	:df: pandas dataframe from which data will be sampled.
	:strata: list containing columns that will be used in the stratified sampling.
	:size: sampling size. If not informed, a sampling size will be calculated
		using Cochran adjusted sampling formula:
		cochran_n = (Z**2 * p * q) /e**2
		where:
			- Z is the z-value. In this case we use 1.96 representing 95%
			- p is the estimated proportion of the population which has an
				attribute. In this case we use 0.5
			- q is 1-p
			- e is the margin of error
		This formula is adjusted as follows:
		adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
		where:
			- cochran_n = result of the previous formula
			- N is the population size
	:seed: sampling seed
	:keep_index: if True, it keeps a column with the original population index indicator
	
	Returns
	-------
	A sampled pandas dataframe based in a set of strata.
	Examples
	--------
	>> df.head()
		id  sex age city 
	0	123 M   20  XYZ
	1	456 M   25  XYZ
	2	789 M   21  YZX
	3	987 F   40  ZXY
	4	654 M   45  ZXY
	...
	# This returns a sample stratified by sex and city containing 30% of the size of
	# the original data
	>> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
	Requirements
	------------
	- pandas
	- numpy
	'''
	population = len(df)
	size = __smpl_size(population, size)
	tmp = df[strata]
	tmp['size'] = 1
	tmp_grpd = tmp.groupby(strata).count().reset_index()
	tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

	# controlling variable to create the dataframe or append to it
	first = True 
	for i in range(len(tmp_grpd)):
		# query generator for each iteration
		qry=''
		for s in range(len(strata)):
			stratum = strata[s]
			value = tmp_grpd.iloc[i][stratum]
			n = tmp_grpd.iloc[i]['samp_size']

			if type(value) == str:
				value = "'" + str(value) + "'"
			
			if s != len(strata)-1:
				qry = qry + stratum + ' == ' + str(value) +' & '
			else:
				qry = qry + stratum + ' == ' + str(value)
		
		# final dataframe
		if first:
			stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
			first = False
		else:
			tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
			stratified_df = stratified_df.append(tmp_df, ignore_index=True)
	
	return stratified_df



def stratified_sample_report(df, strata, size=None):
	'''
	Generates a dataframe reporting the counts in each stratum and the counts
	for the final sampled dataframe.
	Parameters
	----------
	:df: pandas dataframe from which data will be sampled.
	:strata: list containing columns that will be used in the stratified sampling.
	:size: sampling size. If not informed, a sampling size will be calculated
		using Cochran adjusted sampling formula:
		cochran_n = (Z**2 * p * q) /e**2
		where:
			- Z is the z-value. In this case we use 1.96 representing 95%
			- p is the estimated proportion of the population which has an
				attribute. In this case we use 0.5
			- q is 1-p
			- e is the margin of error
		This formula is adjusted as follows:
		adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
		where:
			- cochran_n = result of the previous formula
			- N is the population size
	Returns
	-------
	A dataframe reporting the counts in each stratum and the counts
	for the final sampled dataframe.
	'''
	population = len(df)
	size = __smpl_size(population, size)
	tmp = df[strata]
	tmp['size'] = 1
	tmp_grpd = tmp.groupby(strata).count().reset_index()
	tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
	return tmp_grpd


def __smpl_size(population, size):
	'''
	A function to compute the sample size. If not informed, a sampling 
	size will be calculated using Cochran adjusted sampling formula:
		cochran_n = (Z**2 * p * q) /e**2
		where:
			- Z is the z-value. In this case we use 1.96 representing 95%
			- p is the estimated proportion of the population which has an
				attribute. In this case we use 0.5
			- q is 1-p
			- e is the margin of error
		This formula is adjusted as follows:
		adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
		where:
			- cochran_n = result of the previous formula
			- N is the population size
	Parameters
	----------
		:population: population size
		:size: sample size (default = None)
	Returns
	-------
	Calculated sample size to be used in the functions:
		- stratified_sample
		- stratified_sample_report
	'''
	if size is None:
		cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
		n = round(cochran_n/(1+((cochran_n -1) /population)))
	elif size >= 0 and size < 1:
		n = round(population * size)
	elif size < 0:
		raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
	elif size >= 1:
		n = size
	return n








if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	master = ctreader.mastersheet()

	bins = [12,24,36] # for age in months
	conditions = ['wt', 'het', 'hom']

	#remove genotypes injected or mosaic etc
	master = master[master['genotype'].isin(conditions)]
	
	master = stratified_sample(master, ['age', 'genotype'], size=25, seed = 69)
	master.to_csv('output/otoliths2seg.csv')
	print(master)

	# Group all ages into 6, 12, 24, 36 months old
	master.age = pd.cut(master.age, bins = len(bins), labels=bins, right = False)
	ages= [ctreader.trim(master, 'age', b) for b in bins]

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

	lx= len(data[0])            # Work out matrix dimensions
	ly= len(data[:,0])
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
	ax.set_zlabel('Occurrence', fontsize=20, labelpad=20)
	plt.show()

	






