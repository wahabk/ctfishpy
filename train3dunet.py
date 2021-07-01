import ctfishpy
import segmentation_models as sm
import numpy as np
import gc

#256 mariel needs to be redone
wahab_samples 	= [78,200,240,330,337,341]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,464,364,385]
auto = [41,43,44,45,46,56,57,69,70,72,74,77,78,79,80,90,92,200,201,203]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589], 277
sample = wahab_samples+mariel_samples+zac_samples+auto
val_sample = [40,461,218,462]


unet = ctfishpy.Unet3D('Otoliths')
unet.weightsname = 'test_tv'
unet.comment = 'test_tv'
# unet.rerun = True
unet.lr = 1e-5
unet.batch_size = 1
unet.epochs = 200
unet.alpha = 0.7
unet.train(sample, val_sample)
unet.makeLossCurve()
gc.collect()

