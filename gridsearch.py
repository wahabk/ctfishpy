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
	#cartesian product for grid search
	grid = [dict(zip(keys, v)) for v in itertools.product(*values)]


	# TODO use code here to train model inside this function
	# from https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
	# for v in itertools.product(*values):
	# experiment = dict(zip(keys, v))
	return grid

good_auto = [41,43,44,45,46,56,57,79,80,201,203] # these are good segs from 2d unet
wahab_samples 	= [78,200,218,240,277,330,337,341,462,464,364,385]
mariel_samples	= [421, 423,242,463,259,459,461]
zac_samples		= [257,443,218,364,464]
# 256 mariel needs to be redone, 
# removing 527, 530, 582, 589
# 421 is barx1
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [40]

params = {
	'epochs' : [200],
	'alpha' : [0.6,0.7,0.8,0.9],
	'batch_size' : [1],
	'lr'		: [1e-5],
	'encoder_freeze': [False]
}

grid = genGridSearchParams(params)

print(f'n grids: {len(grid)}')
print(grid)

for g in grid:
	unet = ctfishpy.Unet3D('Otoliths')
	unet.weightsname = 'grdsrch_3d'
	unet.comment = 'grdsrch_3d'
	for key in g:
		setattr(unet, key, g[key])
	unet.train(sample, val_sample)
	unet.makeLossCurve()
	del unet
	gc.collect()
