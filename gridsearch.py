import ctfishpy
import segmentation_models as sm
import numpy as np
import itertools
import gc
from tensorflow.keras import backend as K

def genGridSearchParams(params):
	'''
	Generate series of params
	'''
	
	keys, values = zip(*params.items())
	#cartesian product for grid search
	grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

	# experiment = dict(zip(keys, v))
	return grid

good_auto = [41,43,44,45,46,56,57,79,80,201,203] # these are good segs from 2d unet
wahab_samples 	= [464,364,385, 337,462]
mariel_samples	= [421,423,242,463,259,459,461,530,589]
zac_samples		= [257,443,218,364,464]
# 256 mariel needs to be redone, 
sample = wahab_samples+mariel_samples+zac_samples+good_auto
val_sample = [40,527,582,78,240,277,330,341] #40 527 582 from zack rest from me
test_sample = [527,582]

params = {
	'epochs' : [200],
	'alpha' : [0.1,0.2,0.9],
	'batch_size' : [1],
	'lr'		: [3e-5],
	'BACKBONE' : ['vgg16', 'resnet18'],
	'encoder_freeze': [False, True],
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
	unet.train(sample[:], val_sample[:], test_sample=test_sample[:])
	unet.makeLossCurve()
	del unet
	K.clear_session()
	gc.collect()
