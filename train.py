import ctfishpy


# cc doesnt work on 464 385 421 423 
#256 mariel needs to be redone
wahab_samples 	= [78,200,240,277,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,461]
# removing 527, 530, 582, 589
sample = wahab_samples+mariel_samples
val_samples = [464,364,385,40,218]


unet = ctfishpy.Unet('Otoliths')
unet.rerun = True
unet.epochs= 30

unet.lr = 1e-5
unet.train(sample, val_samples, test_sample=zac_samples)

unet.makeLossCurve()


