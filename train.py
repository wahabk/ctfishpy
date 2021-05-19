import ctfishpy
import segmentation_models as sm
import numpy as np


#256 mariel needs to be redone
wahab_samples 	= [78,200,240,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,218,464,364,385]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589], 277
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [40,461]


unet = ctfishpy.Unet('Otoliths')
unet.weightsname = '2dutr'
unet.comment = '2dutr'
unet.rerun = False
unet.batch_size = 32
unet.lr = 1e-5
unet.alpha = 0.8
unet.epochs = 100
unet.train(sample, val_sample)
unet.makeLossCurve()
del unet



