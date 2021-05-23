import ctfishpy
import segmentation_models as sm
import numpy as np


#256 mariel needs to be redone, 421 is barx1 464 385 bad loc
wahab_samples 	= [78,200,240,330,337,341,462]
mariel_samples	= [423,242,463,259,459]
zac_samples		= [257,443,218,364]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589], 277
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [40,461]


unet = ctfishpy.Unet('Otoliths')
unet.weightsname = 'new_roi'
unet.comment = 'new_roi'
unet.rerun = False
unet.encoder_freeze = True
unet.batch_size = 32
unet.lr = 1e-5
unet.alpha = 0.5
unet.epochs = 200
unet.train(sample, val_sample)
unet.makeLossCurve()
del unet



