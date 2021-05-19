import ctfishpy
import segmentation_models as sm
import numpy as np
import itertools
import gc

def genGridSearchParams(params):
	'''
	Generate series of params
	'''
	
	keys, values = zip(*params.items())
	grid = [dict(zip(keys, v)) for v in itertools.product(*values)]


	# TODO use code here to train model inside this function
	# from https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
	# for v in itertools.product(*values):
	# experiment = dict(zip(keys, v))
	return grid

wahab_samples 	= [78,200,240,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,218,464,364,385]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589], 277
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [40,461]

params = {
	'epochs' : [100],
	'alpha' : [0.9],
	'batch_size' : [16],
	'lr'		: [1e-5],
	'encoder_freeze': [False]
}

grid = genGridSearchParams(params)

print(f'n grids: {len(grid)}')
print(grid)

for g in grid:
	unet = ctfishpy.Unet('Otoliths')
	unet.weightsname = 'test'
	unet.comment = 'grdsrch_utr'
	for key in g:
		setattr(unet, key, g[key])
	unet.roiZ = 150
	unet.train(sample, val_sample)
	unet.makeLossCurve()
	del unet
	gc.collect()
