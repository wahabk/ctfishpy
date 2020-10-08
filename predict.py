from ctfishpy.unet.model import *
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
	model = unet.get_unet()  #unet()
	
	test = testGenie(218, batch_size=16) 
	results = model.predict(test, 16, verbose = 1) # read about this one
	results = fixFormat(results)
	print(results.shape, np.amax(results))
	plt.imsave('first_prediction.png', results[50])
	# lumpfish = ctfishpy.Lumpfish()
	# lumpfish.write_label('prediction.hdf5', results)