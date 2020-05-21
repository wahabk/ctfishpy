import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random

ctreader = ctfishpy.CTreader()
random.seed(a = 111)
#labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'
#label = ctreader.read_labels(labelpath)
#ctreader.view(label[0])


#ct, stack_metadata = ctreader.read(40, r = None)#(1400,1600))
#label = ctreader.read_label(labelpath)


m = ctreader.mastersheet()
wt = ctreader.trim(m, 'genotype', 'wt')
numbers = ctreader.list_numbers(wt)
sample = random.choices(numbers, k=10)
sample.sort()
sample.pop()
print(sample)

sample = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]


for fish in sample:
	print(fish)
	ct, stack_metadata = ctreader.read(fish)
	ctreader.view(ct)

#366 is hires