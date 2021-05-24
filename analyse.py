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
	datapath = 'output/otolithddata.csv'

	nums = ctreader.fish_nums
	nums = nums[:nums.index(423)+1]
	print(nums)

	with open(datapath, 'r') as fr:
		d = json.load(fr)

	for n in nums:
		print(n)

		ct, stack_metadata = ctreader.read(n, align=True)
		label = ctreader.read_label(segs, n, align = False, is_amira=False)

		densities = getDens(ct, label, stack_metadata, nclasses)
		volumes = getVol(label, stack_metadata, nclasses)
		print(densities, volumes)
		data[n] = {'densities': list(densities), 'vols': list(volumes)}

		with open(datapath, 'w') as f:
			json.dump(data, f, sort_keys=True, indent=4)

	
	print(data)
