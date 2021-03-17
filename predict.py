import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
unet.weightspath = 'output/Model/bright_aug_1-10%.hdf5'
n = 582


label = unet.predict(n)
ct, metadata = ctreader.read(n, align=True)

print(label.shape, np.max(label))

ctreader.make_gif(ct[1000:1300], 'output/prediction_new.gif', label = label[1000:1300], fps=20, scale = 100)
