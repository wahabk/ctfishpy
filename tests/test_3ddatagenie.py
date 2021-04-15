import ctfishpy
import os
import time
import numpy as np
timestr = time.strftime("%Y%m%d-%H%M%S")

def fixFormat(batch, label = False):
	# change format of image batches to make viewable with ctreader
	return np.squeeze(batch.astype('uint16'), axis = 3)


if __name__ == "__main__":

	data_gen_args = dict(zoom_range=0.1, # up to 1
				horizontal_flip=True,
				vertical_flip = True,
				# brightness_range = [0.01,0.05],
				fill_mode='constant',
				cval = 0)

				# zoom_range=0.1,
				# 	horizontal_flip=True,
				# 	vertical_flip = True,
				# 	fill_mode='constant',
				# 	cval = 0)

	wahab_samples 	= [78,200,218,240,277,330,337,341,462]
	mariel_samples	= [421,423,242,463,259,459]
	zac_samples		= [257,443,461]
	# removing 527, 530, 582, 589
	val_samples = [464,364,385,40]
	sample = [582,589]
	
	batch_size = 1
	# change label path to read labels directly
	unet = ctfishpy.model.Unet3D('Otoliths')
	unet.steps_per_epoch = 2
	unet.shape = [128,128,128]
	ctreader = ctfishpy.CTreader()

	datagenie = unet.dataGenie(batch_size = batch_size,
							data_gen_args = data_gen_args,
							fish_nums = sample,
							shuffle=True)
	# import pdb; pdb.set_trace()
	while True:
		xdata, ydata  = next(datagenie) #datagenie[0]
		xdata, ydata = xdata[0], ydata[0]
		print('x',xdata.shape, np.max(xdata), np.min(xdata))
		print('y',ydata.shape, np.max(ydata), np.min(ydata))

		label = np.zeros(ydata.shape[:-1], dtype = 'uint8')
		for i in range(3):
			y = ydata[:, :, :, i]
			label[y==1] = i
		print(np.unique(label))

		xdata = fixFormat(xdata*65535)  # remove last weird axis
		print(xdata.shape, np.max(xdata), np.min(xdata))
		print(ydata.shape, np.max(ydata), np.min(ydata))
		#ydata = fixFormat(ydata, label = True)  # remove last weird axis
		# f = input('wait')
		#ctreader.make_gif(xdata[:30], 'output/datagenie.gif', fps=3, label = label[:30])
		ctreader.view(xdata, label)

	#break