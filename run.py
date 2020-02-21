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
from pathlib2 import Path
'''
path = Path('../../Data/HDD/uCT/low_res/EK61_67/EK61_67_[tif]/')

files = sorted(path.iterdir())

images = [str(f) for f in files if f.suffix == '.tif']
#for i in images: print(str(i))
print(images)


'''
c = CTreader()

for i in range(0,65):
	print(i)
	ct, stack_metadata = c.read_dirty(i, r = (0,5))
	print(stack_metadata)

