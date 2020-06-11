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


ct, stack_metadata = ctreader.read(40)
thresh = thresh_stack(ct, 150)
# ctreader.view(thresh)
x, y, z = get_max_projections(thresh)
aspects40 = np.array([x, y, z])
# plot_list_of_3_images(aspects40)
ct = None
gc.collect()

ct, stack_metadata = ctreader.read(41)
thresh41 = thresh_stack(ct, 150)
# ctreader.view(thresh41)
x2 ,y2 ,z2 = get_max_projections(thresh41)
aspects41 = np.array([x2, y2, z2])
# plot_list_of_3_images(aspects41)

temp = aspects40[0]
query = aspects41[0]


ct = None
gc.collect()

#Find Keypoints
print('initialising sift')
temp_kp, descriptors_temp = ctfishpy.pysift.computeKeypointsAndDescriptors(temp)
query_kp, descriptors_query = ctfishpy.pysift.computeKeypointsAndDescriptors(query)
print('working')


#Run stuff
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_temp, descriptors_query, 2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 2

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([temp_kp [m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([query_kp [m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h, w = temp.shape # changed from shape[:-1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)

    query = cv2.polylines(query,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

#Make lines
draw_params = dict(matchColor = (0,0,255), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

temp = to8bit(temp)
query = to8bit(query)

#Draw the lines
img3 = cv2.drawMatches(temp, temp_kp, query, query_kp, good, None, **draw_params)
cv2.imshow('SIFT matches', img3)
cv2.waitKey()


def saveJSON(nparray, jsonpath):
    json.dump(nparray, codecs.open(jsonpath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

def readJSON(jsonpath):
    obj_text = codecs.open(jsonpath, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text)
    return np.array(obj)

tfmpath = 'output/sift_tfm.json'
saveJSON(M.tolist(), tfmpath)
tfm = readJSON(tfmpath)


print('warping perspectives')
new_x = cv2.warpPerspective(aspects41[0], M, (500,500)) 
ctreader.view(new_x)

