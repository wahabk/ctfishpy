import CTFishPy.utility as utility
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
pd.set_option('display.max_rows', None)

#master['age'].value_counts()

CTreader = CTreader()

'''
master = CTreader.mastersheet()

index = utility.findrows(master, 'age', 12)

oneyearolds = utility.trim(master, 'age', 12)

print(oneyearolds)
'''


ct, color = CTreader.read_dirty(0)
#CTreader.view(ct)



circle = CTreader.find_tubes(color[99])


if circle.any():
	cv2.imshow('output', circle)
	cv2.waitKey()

'''
detect 1st fish cap
'''



'''
crop circles to save as single fish
# given x,y are circle center and r is radius
rectX = (x - r) 
rectY = (y - r)
crop_img = self.img[y:(y+2*r), x:(x+2*r)]
'''