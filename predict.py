from ctfishpy.model.dataGenie import *
import ctfishpy
import os
import time
timestr = time.strftime("%Y-%m-%d")
import datetime
import matplotlib.pyplot as plt
import numpy as np
#os.environ[from keras.layers import Input, Conv2D
# from keras.models import Model"CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import Input, Conv2D
from keras.models import Model
from segmentation_models import Unet

if __name__ == "__main__":
	
	
	base_model = Unet('resnet34', encoder_weights=None, input_shape=(128, 128, 3), classes=1, activation='sigmoid')
	inp = Input(shape=(None, None, 1))
	l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
	out = base_model(l1)
	model = Model(inp, out, name=base_model.name)
	pretrained_weights = 'output/Model/unet_checkpoints.hdf5'
	model.load_weights(pretrained_weights)
	
	test = testGenie(88, batch_size=16)  
	results = model.predict(test, 16) # read about this one
	print(results)
	results[results > 0.00001] = 1
	# results[results < 0.5] = 0
	results = np.squeeze(results.astype('float32'), axis = 3)
	test = np.squeeze(test.astype('int8'), axis = 3)
	print(results.shape, np.amax(results), np.mean(results))
	cv2.imwrite('prediction.png', results[10])
	plt.imsave('output/Images/second_prediction2.png', results[20])
	plt.imsave('output/Images/second_prediction_x2.png', test[20])
	ctreader = ctfishpy.CTreader()
	ctreader.view(test, results)
	# lumpfish = ctfishpy.Lumpfish()
	# lumpfish.write_label('prediction.hdf5', results)

