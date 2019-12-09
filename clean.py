import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
from utility import IndexTracker

def readCT(name, direc):
	pass

ct = []
slices_to_read = 250
for i in tqdm(range(1500,1999)):
	ct.append(cv2.imread('../../Data/uCT/EK_208_215/EK_208_215_'+(str(i).zfill(4))+'.tif', cv2.IMREAD_GRAYSCALE))
ct = np.array(ct)
#ct = np.moveaxis(ct, 0, -1)

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, ct.T)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()








