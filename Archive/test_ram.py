import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
random.seed(a = 111)

ctreader = ctfishpy.CTreader()

labelpath = lambda f : f'../../Data/HDD/uCT/Labels/Otolith1/{f}.h5'
sample = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]

for fish in sample:
	ct, stack_metadata = ctreader.read(fish, r = (500, 1500))
	label = ctreader.read_label(labelpath(fish))
	label = label[500:1500]
	ctreader.view(ct, label = label)
	label = None
	ct = None
	gc.collect()

ct1, stack_metadata = ctreader.read(40)

ct2, stack_metadata = ctreader.read(41)
print(type(ct2))
ct3, stack_metadata = ctreader.read(42)
ct4, stack_metadata = ctreader.read(43)
ct5, stack_metadata = ctreader.read(44)
ct6, stack_metadata = ctreader.read(45)
ctreader.view(ct6)
ct7, stack_metadata = ctreader.read(46)
ctreader.view(ct7)

ct8, stack_metadata = ctreader.read(47)
ctreader.view(ct8)