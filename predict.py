import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightspath = 'output/Model/unet_checkpoints.hdf5'
n = 589
label = unet.predict(n)
ct, metadata = ctreader.read(n, align=True)

print(label.shape, np.max(label))

ctreader.make_gif(ct[1200:1500], 'output/prediction_new.gif', label = label[1200:1500], fps=20, scale = 100)
