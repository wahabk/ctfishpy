import ctfishpy



wahab_samples 	= [200,218,240,277,330,337,341,462,464,40,385]
mariel_samples	= [242,256,259,421,423,459,463,530,589]
zac_samples		= [257,443,461,527,582]
sample = wahab_samples+mariel_samples

unet = ctfishpy.model.Unet('Otoliths', sample, zac_samples)
unet.train()
unet.makeLossCurve()


