import ctfishpy
import os
import time
import numpy as np
timestr = time.strftime("%Y%m%d-%H%M%S")

def fixFormat(batch, label = False):
	# change format of image batches to make viewable with ctreader
	if not label: return np.squeeze(batch.astype('uint16'), axis = 3)
	if label: return np.squeeze(batch.astype('uint8'), axis = 3)

if __name__ == "__main__":
	data_gen_args = dict(rotation_range=1, # degrees
				width_shift_range=1, #pixels
				height_shift_range=1,
				shear_range=1, #degrees
				zoom_range=0.1, # up to 1
				horizontal_flip=True,
				vertical_flip = True,
				fill_mode='constant',
				cval = 0)
	wahab_samples 	= [78,200,218,240,277,330,337,341,462]
	mariel_samples	= [421,423,242,463,259,459]
	zac_samples		= [257,443,461] 
	# removing 527, 530, 582, 589
	val_samples = [464,364,385,40]
	sample = val_samples
	
	batch_size = 32
	# change label path to read labels directly
	unet = ctfishpy.model.Unet('Otoliths')
	ctreader = ctfishpy.CTreader()

	xdata, ydata, sample_weights = unet.dataGenie(  batch_size = batch_size,
							data_gen_args = data_gen_args,
							fish_nums = sample,
							shuffle=True)

	label = np.zeros(ydata.shape[:-1], dtype = 'uint8')
	for i in range(3):
		y = ydata[:, :, :, i]
		label[y==1] = i

	print(xdata.shape, np.max(xdata))
	print(ydata.shape, np.max(ydata))
	xdata = fixFormat(xdata*65535)  # remove last weird axis
	#ydata = fixFormat(ydata, label = True)  # remove last weird axis

	ctreader.make_gif(xdata[:30], 'output/datagenie.gif', fps=3, label = label[:30])
	ctreader.view(xdata, label)

	#break