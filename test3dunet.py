import ctfishpy
import segmentation_models as sm
import numpy as np

wahab_samples 	= [78,200,240,277,330,337,341,462,589]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,218]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589]
sample = wahab_samples+mariel_samples
val_sample = [464,364,385,40,461]

unet = ctfishpy.Unet3D('Otoliths')
unet.weightsname = 'bright_aug_1-10%'
unet.comment = 'bright_aug_1-10%'
unet.lr = 1e-5
unet.epochs = 200
unet.train(sample[:1], val_sample[:1], zac_samples[:1])
unet.makeLossCurve()