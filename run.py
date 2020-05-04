import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
random.seed(a = 111)

ctreader = ctfishpy.CTreader()

labelpath = '../../Data/HDD/uCT/Labels/Otolith1/40.h5'
label = ctreader.read_label(labelpath)
ctreader.view(label)


#ct, stack_metadata = ctreader.read(40, r = None)#(1400,1600))
#label = ctreader.read_label(labelpath)

sample = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]


for fish in sample:
	print(fish)
	ct, stack_metadata = ctreader.read(fish)
	ctreader.view(ct)

#test commit