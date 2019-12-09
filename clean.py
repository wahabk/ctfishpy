import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
from utility import IndexTracker

def readCT(name, direc):
	pass

ct = []
slices_to_read = 250
for i in tqdm(range(1800,1900)):
	x = cv2.imread('../../Data/uCT/EK_208_215/EK_208_215_'+(str(i).zfill(4))+'.tif')
	x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	x = cv2.GaussianBlur(x, (5,5), cv2.BORDER_DEFAULT)
	ret, x = cv2.threshold(x, 50, 100, cv2.THRESH_BINARY) #+cv2.THRESH_OTSU)
	ct.append(x)

ct = np.array(ct)

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, ct.T)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

