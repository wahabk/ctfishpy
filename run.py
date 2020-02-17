from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.mainwindowcircle import detectTubes
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
import csv
from qtpy.QtCore import QSettings


xtekctpath = '/home/ak18001/Data/HDD/uCT/low_res/EK61_67/EK61_67.xtekct'

'''
with open(xtekctpath) as f:
    xtekct = json.load(f)
'''

xtekct = QSettings(xtekctpath, QSettings.IniFormat)

print(xtekct.value('XTekCT/VoxelSizeX'))
print(xtekct.value('XTekCT/VoxelSizeY'))
print(xtekct.value('XTekCT/VoxelSizeZ'))


