import ctfishpy
import os
import time
import numpy as np
timestr = time.strftime("%Y%m%d-%H%M%S")

def fixFormat(batch, label = False):
	# change format of image batches to make viewable with ctreader
	return np.squeeze(batch.astype('uint16'), axis = 3)

if __name__ == "__main__":
	data_gen_args = dict(rotation_range=5, # degrees
				width_shift_range=5, #pixels
				height_shift_range=5,
				shear_range=5, #degrees
				zoom_range=0.1, # up to 1
				horizontal_flip=True,
				vertical_flip = True,
				# brightness_range = [0.01,0.05],
				fill_mode='constant',
				cval = 0) 
	wahab_samples 	= [78,200,218,240,330,337,341,462]#277
	mariel_samples	= [421,423,242,463,259,459]
	zac_samples		= [257,443,461]
	# removing 527, 530, 582, 589
	val_samples = [364,40]
	sample = val_samples
	
	batch_size = 32
	# change label path to read labels directly
	unet = ctfishpy.model.Unet('Otoliths')
	unet.steps_per_epoch = 2
	ctreader = ctfishpy.CTreader()

	datagenie = unet.dataGenie(batch_size = batch_size,
							data_gen_args = data_gen_args,
							fish_nums = sample,
							shuffle=True)

	for j in range(2):
		# try: 
		# 	xdata, ydata  = datagenie[0]
		# except:
		xdata, ydata  = next(datagenie)
		print('y',ydata.shape, np.max(ydata), np.min(ydata))
		print('x',xdata.shape, np.max(xdata), np.min(xdata))

		label = np.zeros(ydata.shape[:-1], dtype = 'uint8')
		for i in range(4):
			print(i)
			y = ydata[:, :, :, i]
			label[y==1] = i

		xdata = fixFormat(xdata*65535)  # remove last weird axis
		print(xdata.shape, np.max(xdata), np.min(xdata))
		#ydata = fixFormat(ydata, label = True)  # remove last weird axis

		ctreader.make_gif(xdata[:30], f'output/Model/datagenie_Test/datagenie_{j}.gif', fps=3, label = label[:30])
		# ctreader.view(xdata, label)

