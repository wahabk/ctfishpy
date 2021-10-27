import ctfishpy
import numpy as np
import gc


good_auto = [41,43,44,45,46,56,57,79,80,201,203] # these are good segs from 2d unet
wahab_samples 	= [78,200,218,240,277,330,337,341,462,464,364,385]
mariel_samples	= [421,423,242,463,259,459,461]
zac_samples		= [257,443,218,364,464]
# 256 mariel needs to be redone, 
# removing 527, 530, 582, 589
# 421 is barx1
sample = wahab_samples+mariel_samples+zac_samples
val_sample = [40]


unet = ctfishpy.Unet3D('Otoliths')
unet.weightsname = 'final3d'
unet.comment = 'final3d'
# unet.rerun = True
unet.lr = 1e-5
unet.batch_size = 1
unet.epochs = 180
unet.alpha = 0.6
unet.train(sample[:4], val_sample[:4])
unet.makeLossCurve()
gc.collect()

