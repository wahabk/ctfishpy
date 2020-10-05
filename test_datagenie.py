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
	data_gen_args = dict(rotation_range=0.001,
						width_shift_range=0.01,
						height_shift_range=0.01,
						shear_range=0.01,
						zoom_range=0.3,
						horizontal_flip=True,
						vertical_flip = True,
						fill_mode='constant',
						cval = 0)
	batch_size =16

	sample = [76, 40, 81, 85, 88, 218, 222, 236, 425]
	# change label path to read labels directly

	datagenie = ctfishpy.dataGenie(  batch_size = batch_size,
							data_gen_args = data_gen_args,
							fish_nums = sample)

	ctreader = ctfishpy.CTreader()

	for x_batch, y_batch in datagenie:
		#print(x_batch.shape)
		#print(y_batch.shape)
		x_batch = fixFormat(x_batch)  # remove last weird axis
		y_batch = fixFormat(y_batch, label = True)  # remove last weird axis

		ctreader.view(x_batch, label = y_batch)

		#break

