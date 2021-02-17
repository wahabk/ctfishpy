import ctfishpy
import matplotlib.pyplot as plt
import numpy as np

ctreader = ctfishpy.CTreader()
unet = ctfishpy.Unet('Otoliths')
history = unet.loadHistory('output/Model/History/2021-02-15-23-28_history.json')
unet.makeLossCurve(history)
