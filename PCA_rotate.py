import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import json, codecs
from math import atan2, cos, sin, sqrt, pi

ctreader = ctfishpy.CTreader()

def to8bit(stack):
	if stack.dtype == 'uint16':
		new_stack = ((stack - stack.min()) / (stack.ptp() / 255.0)).astype(np.uint8) 
		return new_stack
	else:
		print('Stack already 8 bit!')
		return stack

def thresh_stack(stack, thresh_8):
	'''
	Threshold CT stack in 16 bits using numpy because it's faster
	provide threshold in 8bit since it's more intuitive then convert to 16
	'''

	thresh_16 = thresh_8 * (65535/255)

	thresholded = []
	for slice_ in stack:
		new_slice = (slice_ > thresh_16) * slice_
		thresholded.append(new_slice)
	return np.array(thresholded)

def get_max_projections(stack):
	'''
	return x, y, x which represent axial, saggital, and coronal max projections
	'''
	x = np.max(stack, axis=0)
	y = np.max(stack, axis=1)
	z = np.max(stack, axis=2)
	return x, y, z

def plot_list_of_3_images(list):
	w=3
	h=1
	fig=plt.figure(figsize=(1, 3))
	columns = 3
	rows = 1
	for i in range(1, columns*rows +1):
		img = list[i-1]
		fig.add_subplot(rows, columns, i)
		plt.imshow(img)
	plt.show()
	plt.clf()
	plt.close()

def resize(img, percent=100):
	width = int(img.shape[1] * percent / 100)
	height = int(img.shape[0] * percent / 100)
	return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def saveJSON(nparray, jsonpath):
	json.dump(nparray, codecs.open(jsonpath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

def readJSON(jsonpath):
	obj_text = codecs.open(jsonpath, 'r', encoding='utf-8').read()
	obj = json.loads(obj_text)
	return np.array(obj)

threshold = 150
ct, stack_metadata = ctreader.read(40)
thresh40 = thresh_stack(ct, threshold)
# ctreader.view(thresh)
x, y, z = get_max_projections(ct)
aspects40 = np.array([x, y, z])
# plot_list_of_3_images(aspects40)

ct = None
gc.collect()

ct, stack_metadata = ctreader.read(41)
thresh41 = thresh_stack(ct, threshold)
# ctreader.view(thresh41)
x2 ,y2 ,z2 = get_max_projections(ct)
aspects41 = np.array([x2, y2, z2])
#plot_list_of_3_images(aspects41)

scale = 75
temp = resize(aspects40[0], scale)
query = resize(aspects41[0], scale)
temp = to8bit(temp)
query = to8bit(query)


# cv2.imshow('', temp)
# cv2.waitKey()
# cv2.imshow('', query)
# cv2.waitKey()
# cv2.destroyAllWindows()

ct = None
gc.collect()

print(temp)
print(temp.shape)
fish = [temp, query]



def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]

    return angle


src = temp
_, bw = cv2.threshold(src, 50, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('', bw)
cv2.waitKey()

## [contours]
# Find all the contours in the thresholded image
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        continue

    # Draw each contour only for visualisation purposes
    cv2.drawContours(src, contours, i, (0, 0, 255), 2)
    # Find the orientation of each shape
    print(getOrientation(c, src))
## [contours]

cv2.imshow('output', src)
cv2.waitKey()

