import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json


if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	segs = 'Otoliths_unet2d'
	nclasses = 3
	datapath = 'output/otolithddata.csv'

	with open(datapath, 'r') as fr:
		data = json.load(fr)

	volumes = {key:data[key]['vols'] for key in data}
	densities = {key:data[key]['densities'] for key in data}
	volumes = np.array([value for key,value in volumes.items()])
	densities = np.array([value for key,value in densities.items()])
	volumes[np.isnan(volumes)] = 0
	densities[np.isnan(densities)] = 0
	print(densities)
		
	fig, [ax1, ax2] = plt.subplots(1,2)
	ax1.set_title('vol')
	ax1.boxplot((volumes[:,0], volumes[:,1],volumes[:,2]))

	ax2.set_title('dens')
	ax2.boxplot((densities[:,0], densities[:,1],densities[:,2]))

	plt.show()