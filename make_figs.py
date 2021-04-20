import ctfishpy
import numpy as np
import matplotlib.pyplot as plt

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightsname = 'final'

sample = [527,40,200,218]#[530, 40,256,421,]
labelled = [40]#[200, 240, 256, 259, 330, 341, 385, 421, 443, 461, 463, 527, 582, 78, 218, 242, 257, 277, 337, 364, 40,  423, 459, 462, 464, 530, 589]
for n in labelled:
	ct, metadata = ctreader.read(n, align = True)
	centers = ctreader.manual_centers
	center = centers[str(n)]

	align = True if n in [78,200,218,240,277,330,337,341,462,464,364,385] else False
	label = ctreader.read_label('Otoliths', n, align=align, is_amira=True)
	# print(np.unique(label, return_counts=True))
	# exit()
	label[label==2]=0
	# predlabel = unet.predict(n)

	projections 			= ctreader.read_max_projections(n)
	predprojections 		= ctreader.read_max_projections(n)
	labelprojections 		= ctreader.make_max_projections(label)
	# predlabelprojections 	= ctreader.make_max_projections(predlabel)
	

	for i, p in enumerate(projections):
		p[ labelprojections[i] == 1 ]=[255,0,0]
		p[ labelprojections[i] == 3 ]=[0,0,255]

	# for i, p_pred in enumerate(predprojections):
	# 	p_pred[ predlabelprojections[i] == 1 ]=[255,0,0]
	# 	p_pred[ predlabelprojections[i] == 2 ]=[0,0,255]


	plt.imsave(f'output/Zac&Mariel/{n}_z_manual.png', projections[0])
	plt.imsave(f'output/Zac&Mariel/{n}_y_manual.png', projections[1])
	plt.imsave(f'output/Zac&Mariel/{n}_x_manual.png', projections[2])

	# plt.imsave(f'output/Zac&Mariel/{n}_z_unet.png', predprojections[0])
	# plt.imsave(f'output/Zac&Mariel/{n}_y_unet.png', predprojections[1])
	# plt.imsave(f'output/Zac&Mariel/{n}_x_unet.png', predprojections[2])

