import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
ctreader = ctfishpy.CTreader()
import gc
import json, codecs

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

threshold = 100
ct, stack_metadata = ctreader.read(40)
thresh40 = thresh_stack(ct, threshold)
# ctreader.view(thresh)
x, y, z = get_max_projections(thresh40)
aspects40 = np.array([x, y, z])
# plot_list_of_3_images(aspects40)
ct = None
gc.collect()

ct, stack_metadata = ctreader.read(41)
thresh41 = thresh_stack(ct, threshold)
# ctreader.view(thresh41)
x2 ,y2 ,z2 = get_max_projections(thresh41)
aspects41 = np.array([x2, y2, z2])
# plot_list_of_3_images(aspects41)

scale = 75
temp = resize(aspects40[0], scale)
query = resize(aspects41[0], scale)

cv2.imshow('', temp)
cv2.waitKey()
cv2.imshow('', query)
cv2.waitKey()
cv2.destroyAllWindows()

ct = None
gc.collect()


fish = [temp, query]
#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
fish_pca = PCA(n_components=0.8)
fish_pca.fit(fish)

ax.imshow(fish_pca.components_[i], cmap=”gray”)

tfmpath = 'output/sift_tfm.json'
saveJSON(M.tolist(), tfmpath)
tfm = readJSON(tfmpath)

plt.show()



# tfmpath = 'output/sift_tfm.json'
# saveJSON(M.tolist(), tfmpath)
# tfm = readJSON(tfmpath)

# print('warping perspectives')
# new_x = cv2.warpPerspective(aspects41[0], M, (500,500)) 
# ctreader.view(new_x)