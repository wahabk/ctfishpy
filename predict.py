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
	unet = ctfishpy.Unet3D('Otoliths')
	unet.weightsname = '3d_test'
	nums = ctreader.fish_nums
	# random.shuffle(nums)

	skip = [424,434,405,465,467]
	centers = ctreader.manual_centers
	nums = nums[nums.index(523)+1:]
	col11s = [256, 257, 258, 259, 421, 423, 424, 425, 431, 432, 433, 434, 443, 456, 457, 458, 459, 460, 461, 462, 463, 464, 582, 583, 584, 585, 586, 587, 588, 589]
	weird = [256, 421, 589, 582, 461, 464, 583, 584, 585, 586, 587]
	#reloc 439 and a few, increase pred height
	#432 bad utricles

	for n in [40]:
		if n in skip: continue
		print(n)
		label, ct = unet.predict(n, thresh = 0.3)
		# print(label.shape, np.max(label))

		# label = ctreader.read_label('Otoliths_unet2d', n, align=False, is_amira=False)
		# ct, stack_metadata = ctreader.read(n, align=True)


		center = centers[str(n)]
		z_center = center[0]
		roiZ = 150

		# label = ctreader.read_label('Otoliths_unet2d', n, align = False, is_amira=False)
		label = ctreader.crop_around_center3d(label, (256,256), center, roiZ=roiZ)

		center[0] = 75
		# ct, stack_metadata = ctreader.read(n, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)
		ct = ctreader.crop_around_center3d(ct, (256,256), center, roiZ=roiZ)
		print(ct.shape)

		# ctreader.write_label('Otoliths_unet2d', label, n)
		# tracker.print_diff()
		ctreader.make_gif(ct, 'output/test_labels.gif', fps=30, label = label)
		# ctreader.view(ct,label)
		gc.collect()

