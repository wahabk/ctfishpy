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


plt.imsave('output/Zac/x_')

