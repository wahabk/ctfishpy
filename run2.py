from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(1):
	ct, color = CTreader.read_dirty(i, r=(0,1999))
	output, circles  = CTreader.find_tubes(color[1000])

	CTreader.view(ct) 

	if output.any():
		cv2.imshow('output', output)
		cv2.waitKey()

cropped_cts = CTreader.crop(ct, circles)
for cropped_ct in cropped_cts:
	CTreader.view(cropped_ct)
#cv2.imshow('output', cropped_cts[0][0])
#cv2.waitKey()

