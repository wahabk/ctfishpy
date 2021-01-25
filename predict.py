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
	
	
	base_model = Unet('resnet34', encoder_weights=None, classes=4, activation='softmax', encoder_freeze=True)
	inp = Input(shape=(224, 224, 1))
	l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
	out = base_model(l1)
	model = Model(inp, out, name=base_model.name)
	pretrained_weights = 'output/Model/unet_checkpoints.hdf5'
	model.load_weights(pretrained_weights)
	
	test = testGenie(40)
	testcopy = test
	print(test.shape, np.amax(test), np.mean(test))

	results = model.predict(test, 64) # read about this one
	
	print(results.shape)
	r    = results[:, :, :, 0], results[:, :, :, 1], results[:, :, :, 2], results[:, :, :, 3] # For RGB results

	# results = np.squeeze(results.astype('float32'), axis = 3)
	testcopy = testcopy*65535
	testcopy = np.squeeze(testcopy.astype('uint16'), axis = 3)
	testcopy = testcopy.astype('uint16')
	print(results.shape, np.amax(results), np.mean(results))
	print(testcopy.shape, np.amax(testcopy), np.mean(testcopy))
	plt.imsave('output/Images/second_prediction2.png', results[20])
	plt.imsave('output/Images/second_prediction_x2.png', testcopy[20])
	ctreader = ctfishpy.CTreader()
	newr = []
	for a in r:
		a[a > 0.5] = 1
		newr.append(a)
	r = np.array(newr, dtype='uint8')
	a,b,c,d = r
	ctreader.view(testcopy, label=a)
	ctreader.view(testcopy, label=b)
	ctreader.view(testcopy, label=c)
	ctreader.view(testcopy, label=d)
	# lumpfish = ctfishpy.Lumpfish()
	# lumpfish.write_label('prediction.hdf5', results)

