import ctfishpy


# cc doesnt work on 464 385 421 423 
wahab_samples 	= [337,341,462,464,385,200,218,330]
mariel_samples	= [242,259,459,463,257, 256] # removed
zac_samples		= [443,40,78,423] # removing 527, 530, 582, 589, 421, 240, 461, 277
sample = wahab_samples+mariel_samples
val_samples = zac_samples

unet = ctfishpy.Unet('Otoliths')
unet.rerun = True
unet.epochs= 20
# unet.lr = 1e-6
unet.train(sample, val_samples)

unet.makeLossCurve()


