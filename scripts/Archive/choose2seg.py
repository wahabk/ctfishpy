import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random

ctreader = ctfishpy.CTreader()
random.seed(a = 111)

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