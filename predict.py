import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightsname = '2dutr'

for n in [589]: #ctreader.fishnums:
	label = unet.predict(n)
	ct, metadata = ctreader.read(n, align=True)

	print(label.shape, np.max(label))

	ctreader.write_label('Otoliths_unet2d', label, n)
	# ctreader.make_gif(ct[1200:1500], 'output/test_labels.gif', fps=30, label = label[1200:1500])

