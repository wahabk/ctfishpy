import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightspath = 'output/Model/unet_checkpoints.hdf5'
label, ct = unet.predict(40)
print(label.shape, np.max(label), np.unique(label))

ctreader.make_gif(ct, 'output/prediction.gif', label = label, fps=15, scale = 300)
