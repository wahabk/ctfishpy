import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	centers = ctreader.manual_centers
	segs = 'Otoliths_unet2d'
	datapath = 'output/otolith_data.csv'
	nclasses = 3
	skip = [424,434,435,436,405,465,467] # these dont exist!! but are wt so ok
	col11s = [256, 257, 258, 259, 421, 423, 424, 425, 431, 432, 433, 434, 443, 456, 457, 458, 459, 460, 461, 462, 463, 464, 582, 583, 584, 585, 586, 587, 588, 589]
	roiSize = (200,200,300)

	with open(datapath, 'r') as fr:
		data = json.load(fr)
	done = [int(d) for d in data.keys()]

	for n in ctreader.fish_nums:
		print(n)
		if n in done or n in skip: continue
		center = centers[str(n)]

		label = ctreader.read_label(segs, n, is_amira=False)
		label = ctreader.crop3d(label, roiSize, center=center)

		#only read range
		roiZ = roiSize[0]
		z_center = center[0]
		center[0] = int(roiSize[0]/2)
		print(roiSize, z_center)
		ct, stack_metadata = ctreader.read(n, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)
		ct = ctreader.crop3d(ct, roiSize, center=center)

		print(label.shape, ct.shape)

		densities = ctreader.getDens(ct, label, nclasses)
		volumes = ctreader.getVol(label, stack_metadata, nclasses)
		print(densities, volumes)
		data[str(n)] = {'densities': list(densities), 'vols': list(volumes)}

		with open(datapath, 'w') as f:
			json.dump(data, f, sort_keys=True, indent=4)

	
	print(data)
