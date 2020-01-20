from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
pd.set_option('display.max_rows', None)

CTreader = CTreader()

<<<<<<< HEAD
for i in range(5):
	ct, color = CTreader.read_dirty(5, r=(950,1000))
=======
#CTreader.view(ct)
for i in range(5,12):
	ct, color = CTreader.read_dirty(i, r=(950,1000))
>>>>>>> parent of 5556536... finished circle cropping
	output, circles  = CTreader.find_tubes(color[30])

	if output.any():
		print(circles.shape[0]) # number of circles detected
<<<<<<< HEAD

cropped_cts = utility.crop(ct, circles)
CTreader.view(cropped_cts[0])
cv2.imshow('output', cropped_cts[0][0])
cv2.waitKey()

=======
'''
def crop(self, ct, circles):
	CTs = []

	for circle in circles:
		for slice_ in ct:
			x = slice_.cv.crop()

	pass
	return CTs
'''

# manually label each tube? what if they're at an angle?



'''
crop circles to save as single fish
# given x,y are circle center and r is radius
rectX = (x - r) 
rectY = (y - r)
crop_img = self.img[y:(y+2*r), x:(x+2*r)]
'''
>>>>>>> parent of 5556536... finished circle cropping
