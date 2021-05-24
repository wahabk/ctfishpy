import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import random

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightsname = 'new_roi'
nums = ctreader.fish_nums
# random.shuffle(nums)

for n in nums:
	print(n)
	label, ct = unet.predict(n, thresh = 0.3)
	print(label.shape, np.max(label))

	ctreader.write_label('Otoliths_unet2d', label, n)
	# ctreader.make_gif(ct[1200:1500], 'output/test_labels.gif', fps=30, label = label[1200:1500])
	# ctreader.view(ct,label)

