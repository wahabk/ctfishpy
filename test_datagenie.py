import ctfishpy
import os
import time
import numpy as np
timestr = time.strftime("%Y%m%d-%H%M%S")

def fixFormat(batch, label = False):
	# change format of image batches to make viewable with ctreader
	if not label: return np.squeeze(batch.astype('uint16'), axis = 3)
	if label: 
		batch 
		return np.squeeze(batch.astype('uint8'), axis = 3)

if __name__ == "__main__":
	data_gen_args = dict(rotation_range=30, # degrees
						width_shift_range=20, #pixels
						height_shift_range=20,
						shear_range=20, #degrees
						zoom_range=0.1, # up to 1
						horizontal_flip=True,
						vertical_flip = True,
						fill_mode='constant',
						cval = 0)
	sample = [200]
	batch_size = 32
	# change label path to read labels directly

	datagenie = ctfishpy.dataGenie(  batch_size = batch_size,
							data_gen_args = data_gen_args,
							fish_nums = sample)

	ctreader = ctfishpy.CTreader()
	xdata, ydata, sample_weights = datagenie
	ydata    = ydata[:, :, :, 1]

	print(xdata.shape, np.max(xdata))
	print(ydata.shape, np.max(ydata))
	xdata = fixFormat(xdata*65535)  # remove last weird axis
	#ydata = fixFormat(ydata, label = True)  # remove last weird axis

	ctreader.view(xdata, label = ydata)

	#break