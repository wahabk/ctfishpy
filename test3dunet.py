import ctfishpy
import segmentation_models as sm
import numpy as np

wahab_samples 	= [78,200,240,277,330,337,341,462,589,218]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,218,443,461]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589]
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [464,364,385,40]

unet = ctfishpy.Unet3D('Otoliths')
unet.weightsname = '3d test'
unet.comment = '3d test'
unet.lr = 1e-5
unet.epochs = 400
unet.train(sample, val_sample)
unet.makeLossCurve()

