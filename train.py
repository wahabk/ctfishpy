import ctfishpy
import segmentation_models as sm
import numpy as np


# cc doesnt work on 464 385 421 423, 78
#256 mariel needs to be redone
wahab_samples 	= [78,200,240,277,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443,218]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589]
sample = wahab_samples+mariel_samples
val_sample = [464,364,385,40,461]

unet = ctfishpy.Unet('Otoliths')
unet.weightsname = 'bright_aug_1-10%'
unet.comment = 'bright_aug_1-10%'
unet.lr = 1e-5
unet.epochs = 200
unet.train(sample[:1], val_sample[:1], zac_samples[:1])
unet.makeLossCurve()

# params = [[1e-5, 150, 'cross_entropy', 	0.7, 	[1,1,1],			32, True],
# 			[1e-5, 150, 'dice_loss', 		0.7, 	[1,1,1],			32, True],
# 			[1e-5, 150, 'dice_loss', 		0.7, 	[0.5,1.5,1.5],		32, True],
# 			[1e-5, 150, 'tversky', 			0.7, 	[1,1,1],			32, True],
# 			[1e-5, 150, 'dice_loss', 		0.7, 	[1,1.5,1.5],		32, True],
# 			[1e-5, 150, 'dice_loss', 		0.7, 	[0.5,1.25,1.5],		32, True],
# 			[1e-5, 150, 'dice_loss', 		0.7, 	[1,2,2],			32, True],
# 			[1e-5, 150, 'tversky',			0.8,	[1,1,1],			32, True],
# 			[1e-5, 150, 'tversky',			0.6,	[1,1,1],			32, True],
# 			[1e-5, 150, 'tversky',			0.5,	[1,1,1],			32, True]]


# params = [	[1e-5, 150, 'tversky', 0.5, [1,1,1], 16 ,True],
# 			[1e-5, 150, 'tversky', 0.5, [1,1,1], 10 ,True],
# 			[1e-5, 150, 'tversky', 0.5, [1,1,1], 25 ,True],
# 			[1e-3, 150, 'tversky', 0.5, [1,1,1], 32 ,True],
# 			[1e-4, 150, 'tversky', 0.5, [1,1,1], 32 ,True],
# 			[1e-6, 150, 'tversky', 0.5, [1,1,1], 32 ,True],
# 			[1e-5, 150, 'tversky', 0.5, [1,1,1], 16 ,False]]




# for lr, epochs, loss, alpha, class_weights, batch_size, encoder_freeze in params:
# 	unet = ctfishpy.Unet('Otoliths')
# 	unet.weightsname = 'unet'
# 	unet.comment = 'Mariel_parameters'
# 	unet.lr = lr
# 	unet.epochs = epochs
# 	unet.alpha = alpha
# 	unet.class_weights = np.array(class_weights)
# 	unet.batch_size = batch_size
# 	unet.encoder_freeze = encoder_freeze

# 	if loss == 'tversky': unet.loss = unet.multi_class_tversky_loss
# 	if loss == 'dice_loss': unet.loss = sm.losses.DiceLoss(class_weights=unet.class_weights)
# 	if loss == 'cross_entropy': unet.loss = sm.losses.categorical_crossentropy

# 	unet.train(sample, val_sample, zac_samples)
# 	unet.makeLossCurve()
# 	del unet



