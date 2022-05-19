from matplotlib import pyplot as plt
from CTFishPy.utility import IndexTracker
import argparse
from tqdm import tqdm
import numpy as np
import cv2

ct = []
color = []
slices_to_read = 250

for i in tqdm(range(1800,1900)):
	x = cv2.imread('../../Data/uCT/low_res/EK_208_215/EK_208_215_'+(str(i).zfill(4))+'.tif')
	color.append(x)
	x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	#x = cv2.GaussianBlur(x, (5,5), cv2.BORDER_DEFAULT)
	ret, x = cv2.threshold(x, 50, 100, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	ct.append(x)

ct = np.array(ct)
color = np.array(color)

input_ = color[0].copy()
resize = 0.4
input_ = cv2.resize(input_, None, fx=resize, fy=resize)
output = input_.copy()
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, minRadius=40) #param1=50, param2=30,

print(circles)
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 0, 255), 2)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

	cv2.imshow("output", output)
	cv2.waitKey(0)

else:
	print('No circles found :(')

'''
fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, ct.T)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
save = str(input('Save figure? '))
if save:
	fig.savefig('output/'+save+'.png')
'''