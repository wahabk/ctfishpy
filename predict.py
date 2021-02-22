import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightspath = 'output/Model/unet_checkpoints.hdf5'
label = unet.predict(40)
ct, metadata = ctreader.read(40, align=True)

print(label.shape, np.max(label))

ctreader.make_gif(ct[1300:1500], 'output/prediction_new.gif', label = label[1300:1500], fps=20, scale = 100)
