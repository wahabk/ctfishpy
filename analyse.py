import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json

def getVol(label, metadata, nclasses):
	counts = np.array([np.count_nonzero(label == i) for i in range(1, nclasses+1)])
	voxel_size = float(metadata['VoxelSizeX']) * float(metadata['VoxelSizeY']) * float(metadata['VoxelSizeZ'])
	volumes = counts * voxel_size

	return volumes

def getDens(scan, label, metadata, nclasses):
	dens_calib = 0.0000381475547417411
	voxel_values = np.array([np.mean(scan[label == i]) for i in range(1, nclasses+1)])
	densities = voxel_values * dens_calib

	return densities
	

if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	segs = 'Otoliths_unet2d'
	nclasses = 3
	datapath = 'output/otolithddatacol11.csv'
	centers = ctreader.manual_centers
	nums = ctreader.fish_nums
	nums = nums[:nums.index(423)+1]
	print(nums)

	# with open(datapath, 'r') as fr:
	# 	data = json.load(fr)
	data = {}
	skip = [424,434,405,465,467]
	
	col11s = [256, 257, 258, 259, 421, 423, 424, 425, 431, 432, 433, 434, 443, 456, 457, 458, 459, 460, 461, 462, 463, 464, 582, 583, 584, 585, 586, 587, 588, 589]

	for n in col11s:
		print(n)
		if n in skip: continue
		center = centers[str(n)]
		z_center = center[0]
		roiZ = 150

		label = ctreader.read_label(segs, n, align = False, is_amira=False)
		label = ctreader.crop_around_center3d(label, (256,256), center, roiZ=roiZ)

		center[0] = 75
		ct, stack_metadata = ctreader.read(n, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)
		ct = ctreader.crop_around_center3d(ct, (256,256), center, roiZ=roiZ)

		print(label.shape, ct.shape)

		densities = getDens(ct, label, stack_metadata, nclasses)
		volumes = getVol(label, stack_metadata, nclasses)
		print(densities, volumes)
		data[str(n)] = {'densities': list(densities), 'vols': list(volumes)}

		with open(datapath, 'w') as f:
			json.dump(data, f, sort_keys=True, indent=4)

	
	print(data)
