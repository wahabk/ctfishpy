import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
history = unet.loadHistory('output/Model/History/2021-02-24-08-41_history.json')

print(history.keys())
loss = history['loss']
valf1 = history['val_f1-score']
valf1[loss.index(min(loss))]

print(valf1[loss.index(min(loss))])



# unet.makeLossCurve(history)