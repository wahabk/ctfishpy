import ctfishpy
import numpy as np
import gc

good_auto = [41,43,44,45,46,56,57,79,80,201,203] # these are good segs from 2d unet
wahab_samples 	= [464,364,385, 337,462]
mariel_samples	= [421,423,242,463,259,459,461,530,589]
zac_samples		= [257,443,218,364,464]
# 256 mariel needs to be redone, DONE
sample = wahab_samples+mariel_samples+zac_samples+good_auto
val_sample = [40,527,582,78,240,277,330,341] #40 527 582 from zack rest from me
# test_sample = []

print(len(sample), len(val_sample))

unet = ctfishpy.Unet3D('Otoliths')
unet.weightsname = 'final3d'
unet.comment = 'final3d'
# unet.rerun = True
unet.BACKBONE = 'vgg16'
unet.lr = 3e-5
unet.batch_size = 2
unet.epochs = 80
unet.alpha = 0.7
unet.train(sample, val_sample)
unet.makeLossCurve()
gc.collect()
