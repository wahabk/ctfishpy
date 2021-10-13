import ctfishpy
import numpy as np
import matplotlib.pyplot as plt

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightsname = 'final'
project = 'yushipaper'

sample = [527,40,200,218]#[530, 40,256,421,]
labelled = [40]#[200, 240, 256, 259, 330, 341, 385, 421, 443, 461, 463, 527, 582, 78, 218, 242, 257, 277, 337, 364, 40,  423, 459, 462, 464, 530, 589]

for n in labelled:
	ct, metadata = ctreader.read(n, align = True)
	centers = ctreader.manual_centers
	center = centers[str(n)]

	label = ctreader.read_label('Otoliths', n, is_amira=True)
	projections = ctreader.read_max_projections(n)
	projections = ctreader.label_projections(projections, label)


	# label = unet.predict(n)


	plt.imsave(f'output/{project}/{n}_z_manual.png', projections[0])
	plt.imsave(f'output/{project}/{n}_y_manual.png', projections[1])
	plt.imsave(f'output/{project}/{n}_x_manual.png', projections[2])

	# plt.imsave(f'output/{project}/{n}_z_unet.png', predprojections[0])
	# plt.imsave(f'output/{project}/{n}_y_unet.png', predprojections[1])
	# plt.imsave(f'output/{project}/{n}_x_unet.png', predprojections[2])

