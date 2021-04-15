import ctfishpy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

	roiSize = 224
	thresh = range(10, 130, 10)
	print(thresh)
	labelled = [200, 240, 256, 259, 330, 341, 385, 421, 443, 461, 463, 527, 582, 78, 218, 242, 257, 277, 337, 364, 40,  423, 459, 462, 464, 530, 589]

	#failed on 421, 78, 423,  464, 385
	# failed = [90, 91, 92, 93, 94, 95, 265, 383, 384, 385, 386, 388, 389, 398, 399, 400, 401, 402, 404, 405, 420, 421, 423, 424, 425, 426, 427, 428, 429, 430, 441, 447, 466, 467, 470, 471, 472, 473, 475, 477, 497, 498, 499]
	ctreader = ctfishpy.CTreader()
	template = ctreader.read_label('Otoliths', 0)
	

	# made = [path.stem for path in projectionspath.iterdir() if path.is_file() and path.suffix == '.png']
	# made = [int(name.replace('x_', '')) for name in made]
	# made.sort()


	mancenters = ctreader.manual_centers
	data = {}

	for t in thresh:
		print('thresh: ', t)
		failed = []
		errors = []
		for n in labelled:
			print(f'fish: {n}')
			# ct, metadata = ctreader.read(n, align = True)
			z,y,x = ctreader.read_max_projections(n)
			center = ctfishpy.cc(n, template, t, roiSize)
			
			# try:
			# 	center = ctfishpy.cc(n, template, thresh, roiSize)
			# except:
			# 	print('failed')
			# 	failed.append(n)
			# 	continue
			# if center == [0,0,0]: 
			# 	print('failed')
			# 	failed.append(n)
			# 	continue
			

			if n in labelled:
				mancenter = mancenters[str(n)]
				x_diff = (center[2] - mancenter[2])**2
				y_diff = (center[1] - mancenter[1])**2
				z_diff = (center[0] - mancenter[0])**2
				square_error = x_diff + y_diff + z_diff
				error = int(math.sqrt(square_error))
				print(error)
				errors.append(int(error))

			# zcenter = center[:]
			# xcenter = center[:]
			# zcenter.pop(0)
			# xcenter.pop(-1)
			# print(zcenter, xcenter, center)
			# otolith = ctreader.crop_around_center2d(z, zcenter, roiSize)
			# plt.imshow(otolith)
			# plt.show()
			# otolith = ctreader.crop_around_center2d(x, xcenter, roiSize)
			# plt.imshow(otolith)
			# plt.show()
			# ctreader.view(otolith)
		print(f'failed on {len(failed)/len(ctreader.fish_nums)} which are \n{failed}')
		errors = np.array(errors)
		data[f'{t}'] = errors
		#print(errors, np.mean(errors), np.max(errors), np.min(errors)) 
		# np.savetxt('output/cc_errors.csv', errors)
	
	# data = pd.DataFrame.from_dict(data)
	print(data)


	import pdb; pdb.set_trace()

	names = data.keys()
	vals = data.values()
	plt.boxplot(vals)
	plt.show()



