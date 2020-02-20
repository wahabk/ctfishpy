from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(0,1):
	ct, stack_metadata = CTreader.read_dirty(i, r=(0,1999), scale = 40)
	circle_dict  = CTreader.find_tubes(ct)

	#CTreader.view(ct) 

ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
cropped_cts = CTreader.crop(ct, ordered_circles)

i=0
c = cropped_cts[i]
c = np.mean(np.mean(np.mean(c, axis = 1), axis = 1), axis = 1)
plt.subplot(4, 1, i+1)
plt.plot(np.arange(c.shape[0]), c, color='green')
plt.ylabel("Pixel intensity")
plt.title("Average pixel intensity of each slice 0")

i=1
c = cropped_cts[i]
c = np.mean(np.mean(np.mean(c, axis = 1), axis = 1), axis = 1)
plt.subplot(4, 1, i+1)
plt.plot(np.arange(c.shape[0]), c, color='green')
plt.ylabel("Pixel intensity")

i=2
c = cropped_cts[i]
c = np.mean(np.mean(np.mean(c, axis = 1), axis = 1), axis = 1)
plt.subplot(4, 1, i+1)
plt.plot(np.arange(c.shape[0]), c, color='green')
plt.ylabel("Pixel intensity")

i=3
c = cropped_cts[i]
c = np.mean(np.mean(np.mean(c, axis = 1), axis = 1), axis = 1)
plt.subplot(4, 1, i+1)
plt.plot(np.arange(c.shape[0]), c, color='green')
plt.xlabel("CT slice")
plt.ylabel("Pixel intensity")

plt.show()