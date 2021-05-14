import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightsname = 'test'

for n in ctreader.fishnums:
	label = unet.predict(n)
	ct, metadata = ctreader.read(n, align=True)

	print(label.shape, np.max(label))

	ctreader.write_label('Otoliths_unet2d', label, n)
