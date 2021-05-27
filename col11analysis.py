import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json
import csv

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	master = ctreader.mastersheet()
	datapath = 'output/otolithddatacol11.csv'

	conditions = ['col11a2']
	master = master[master['strain'].isin(conditions)]
	# print(master)
	print(list(master['n']))

	master.to_csv('output/col11master.csv')


	col11s = [256, 257, 258, 259, 421, 423, 424, 425, 431, 432, 433, 434, 443, 456, 457, 458, 459, 460, 461, 462, 463, 464, 582, 583, 584, 585, 586, 587, 588, 589]

	with open(datapath, 'r') as fr:
		data = json.load(fr)

			
	with open('output/otolithddatacol11_2.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
		w = csv.DictWriter(f, ['n', 'vols', 'densities'])
		w.writeheader()
		for k,v in data.items():
			w.writerow(v)

	for n in col11s:
		print(n)

		ct, stack_metadata = ctreader.read(n, align=True)
		label = ctreader.read_label(segs, n, align = False, is_amira=False)

		densities = ctreader.getDens(ct, label, stack_metadata, nclasses)
		volumes = ctreader.getVol(label, stack_metadata, nclasses)
		print(densities, volumes)
		ctreader.view(ct, label)

		# data[n] = {'densities': list(densities), 'vols': list(volumes)}
		# with open(datapath, 'w') as f:
		# 	json.dump(data, f, sort_keys=True, indent=4)

