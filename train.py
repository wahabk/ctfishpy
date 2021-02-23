import ctfishpy


# cc doesnt work on 464 385 421 423, 78
#256 mariel needs to be redone
wahab_samples 	= [78,200,240,277,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,218]
# removing 527, 530, 582, 589
sample = wahab_samples+mariel_samples
val_sample = [464,364,385,40,461]


unet = ctfishpy.Unet('Otoliths')

unet.weightsname = 'Final_Tversky'
unet.comment = 'Final_Tversky'

unet.train(sample, val_sample, test_sample=zac_samples)
unet.makeLossCurve()

unet = None

unet = ctfishpy.Unet('Otoliths')

unet.weightsname = 'Final_Weighted_Dice'
unet.comment = 'Final_Weighted_Dice'

unet.train(sample, val_sample, test_sample=zac_samples)
unet.makeLossCurve()

