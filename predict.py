from ctfishpy.unet.model import *
from ctfishpy.dataGenie import *
import ctfishpy
import os
import time
timestr = time.strftime("%Y-%m-%d")
import datetime
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
	
	unet = Unet()
	model = unet.get_unet()  #unet()
	
	testGenie = testGenie(40)
	results = model.predict(testGenie, 1, verbose = 1) # read about this one
	import pdb; pdb.set_trace()
	plt.imsave()
	# lumpfish = ctfishpy.Lumpfish()
	# lumpfish.write_label('prediction.hdf5', results)