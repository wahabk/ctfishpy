from re import A
from tensorflow.core.framework.types_pb2 import DT_DOUBLE

from tensorflow.python.keras.optimizer_v2.adam import Adam
import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import random
import gc

# checking for memory leaks
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()


if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	unet = ctfishpy.Unet('Otoliths')
	unet.weightsname = 'new_roi'
	nums = ctreader.fish_nums
	# random.shuffle(nums)
	'''
	424 cant read
	434 
	'''

	nums = nums[nums.index(434)+1:]
	for n in nums:
		print(n)
		label, ct = unet.predict(n, thresh = 0.3)
		print(label.shape, np.max(label))

		ctreader.write_label('Otoliths_unet2d', label, n)
		# tracker.print_diff()
		
		# ctreader.make_gif(ct[1200:1500], 'output/test_labels.gif', fps=30, label = label[1200:1500])
		# ctreader.view(ct,label)
		gc.collect()

