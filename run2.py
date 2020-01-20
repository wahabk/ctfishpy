from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
pd.set_option('display.max_rows', None)

CTreader = CTreader()

for i in range(5):
	ct, color = CTreader.read_dirty(5, r=(950,1000))
	output, circles  = CTreader.find_tubes(color[30])

	if output.any():
		print(circles.shape[0]) # number of circles detected

cropped_cts = utility.crop(ct, circles)
CTreader.view(cropped_cts[0])
cv2.imshow('output', cropped_cts[0][0])
cv2.waitKey()

