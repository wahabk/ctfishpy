import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
history = unet.loadHistory('output/Model/History/2021-02-03-10-51_history.json')
unet.makeLossCurve(history)
