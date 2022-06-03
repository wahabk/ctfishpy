import ctfishpy
import numpy as np 

labelled = [200, 240, 256, 259, 330, 341, 385, 421, 443, 461, 463, 527, 582, 78, 218, 242, 257, 277, 337, 364, 40,  423, 459, 462, 464, 530, 589]
ctreader = ctfishpy.CTreader()

distrib = []
for l in labelled:
	label = ctreader.read_label('Otoliths', l, is_amira=True)
	data = np.unique(label, return_counts=True)[1]
	print(data[1:])
	distrib.append(data[1:])

distrib = np.array(distrib)
average = np.mean(distrib, axis=0)
std = np.std(distrib, axis=0)
ratio = average / np.min(average)

print(average, std, ratio)
