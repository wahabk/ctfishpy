from ctfishpy.model.model import *
from ctfishpy.dataGenie import *
import ctfishpy
import os
import time
timestr = time.strftime("%Y-%m-%d")
import datetime
import matplotlib.pyplot as plt
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
	
	unet = Unet()
	model = unet.get_unet(preload=True)  #unet()
	
	test = testGenie(218, batch_size=16) 
	results = model.predict(test, 10) # read about this one
	results = np.squeeze(results.astype('float32'), axis = 3)
	test = np.squeeze(test.astype('float32'), axis = 3)
	print(results.shape, np.amax(results))
	cv2.imwrite('prediction.png', results[10])
	plt.imsave('output/Images/second_prediction.png', results[20])
	plt.imsave('output/Images/second_prediction_x.png', test[20])
	ctreader = ctfishpy.CTreader()
	ctreader.view(test, results)
	# lumpfish = ctfishpy.Lumpfish()
	# lumpfish.write_label('prediction.hdf5', results)

