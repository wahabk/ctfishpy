import ctfishpy
import numpy as np
import matplotlib.pyplot as plt

ctreader = ctfishpy.CTreader()

n=40

projections = ctreader.read_max_projections(n)
label = ctreader.read_label('Otoliths', n)
labelprojections = ctreader.make_max_projections(label)

for i, p in enumerate(projections):
	p[labelprojections[i]==1]=[255,0,0]
	p[labelprojections[i]==2]=[255,255,0]
	p[labelprojections[i]==3]=[0,0,255]


plt.imsave(f'output/Zac&Mariel/{n}_z_labelled.png', projections[0])
plt.imsave(f'output/Zac&Mariel/{n}_y_labelled.png', projections[1])
plt.imsave(f'output/Zac&Mariel/{n}_x_labelled.png', projections[2])

