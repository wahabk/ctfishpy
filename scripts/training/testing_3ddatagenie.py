import ctfishpy
import os
import time
import numpy as np
timestr = time.strftime("%Y%m%d-%H%M%S")

def fixFormat3d(batch):
	# change format of image batches to make viewable with ctreader
	return np.squeeze((batch*65535).astype('uint16'), axis = 3)

if __name__ == "__main__":

	data_gen_args_3d = dict(rotation_range=5, # degrees
				width_shift_range=5, #pixels
				height_shift_range=5,
				shear_range=5, #degrees
				zoom_range=0.1, # up to 1
				horizontal_flip=True,
				vertical_flip = True,
				# brightness_range = [0.01,0.05],
				fill_mode='constant',
				cval = 0) 

	data_gen_args = dict(horizontal_flip=True,
		vertical_flip = True,
		fill_mode='constant',
		#cval = 0,
		)
	
	sample = [40]
	batch_size = len(sample)
	n_classes = 4
	# unet.steps_per_epoch = 2
	unet = ctfishpy.tf_model.Unet3D('Otoliths')
	ctreader = ctfishpy.CTreader()
	datagenie = unet.dataGenie(batch_size = batch_size,
							data_gen_args = data_gen_args,
							fish_nums = sample,
							shuffle=True)


	# for j in range(batch_size):
	# 	# try: 
	# 	# 	xdata, ydata  = datagenie[0]
	# 	# except:
	xdata_batch, ydata_batch  = next(datagenie)
	print('y',ydata_batch.shape, np.max(ydata_batch), np.min(ydata_batch))
	print('x',xdata_batch.shape, np.max(xdata_batch), np.min(xdata_batch))

	for j, (xdata, ydata) in enumerate(zip(xdata_batch, ydata_batch)):
		print('y',ydata.shape, np.max(ydata), np.min(ydata))
		print('x',xdata.shape, np.max(xdata), np.min(xdata))

		label = np.zeros(ydata.shape[:-1], dtype = 'uint8')

		# reverse one hot encoding
		for i in range(n_classes):
			y = ydata[:, :, :, i]
			label[y==1] = i

		xdata = fixFormat3d(xdata)  # remove last weird axis
		print(xdata.shape, np.max(xdata), np.min(xdata))
		#ydata = fixFormat(ydata, label = True)  # remove last weird axis

		ctreader.make_gif(xdata, f'output/test_3ddatagenie_{j}.gif', fps=20, label = label)
		# ctreader.view(xdata, label)

