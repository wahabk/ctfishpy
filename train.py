import ctfishpy



wahab_samples 	= [337,341,462,40,78, 464, 385] # cc doesnt work on 464 385
mariel_samples	= [242,256,259,459,463,530,589,421,423] # 421 423 removed
zac_samples		= [257,443,461,527,582]
sample = wahab_samples+mariel_samples+zac_samples
val_samples= [200,218,240,277,330] # from me

unet = ctfishpy.model.Unet('Otoliths')
unet.rerun = True
unet.train(sample, val_samples)

unet.makeLossCurve()


