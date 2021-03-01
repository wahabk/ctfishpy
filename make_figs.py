import ctfishpy
import numpy as np
import matplotlib.pyplot as plt

ctreader = ctfishpy.CTreader()

n=530

ct, metadata = ctreader.read(n, align = True)
centers = ctreader.manual_centers
center = centers[str(n)]

otolith = ctreader.crop_around_center3d(ct, 200, center)
ctreader.make_gif(otolith, 'output/ncoa3_oto2.gif', fps=20)

# projections = ctreader.read_max_projections(n)
# label = ctreader.read_label('Otoliths', n)
# labelprojections = ctreader.make_max_projections(label)

# for i, p in enumerate(projections):
# 	p[ labelprojections[i] == 1 ]=[255,0,0]
# 	p[ labelprojections[i] == 2 ]=[255,255,0]
# 	p[ labelprojections[i] == 3 ]=[0,0,255]


# plt.imsave(f'output/Zac&Mariel/{n}_z_labelled.png', projections[0])
# plt.imsave(f'output/Zac&Mariel/{n}_y_labelled.png', projections[1])
# plt.imsave(f'output/Zac&Mariel/{n}_x_labelled.png', projections[2])

