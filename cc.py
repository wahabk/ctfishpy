import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
import ctfishpy.pysift as pysift
ctreader = ctfishpy.CTreader()
import gc

def to16bit(stack):
    if stack.dtype == 'uint16':
        new_stack = ((stack - stack.min()) / (stack.ptp() / 255.0)).astype(np.uint8) 
    return new_stack


ct, stack_metadata = ctreader.read(40)

thresh = []
for slice_ in ct:
    new_slice = (slice_ > 38550) * slice_
    thresh.append(new_slice)
thresh = np.array(thresh)

x = np.max(thresh, axis=0)
y = np.max(thresh, axis=1)
z = np.max(thresh, axis=2)

aspects40 = np.array([x, y, z])

w=3
h=1
fig=plt.figure(figsize=(1, 3))
columns = 3
rows = 1
for i in range(1, columns*rows +1):
    img = aspects40[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

ct = None
ct, stack_metadata = ctreader.read(42)

thresh41 = []
for slice_ in ct:
    new_slice = (slice_ > 38550) * slice_
    thresh41.append(new_slice)
thresh41 = np.array(thresh)

x2 = np.max(thresh41, axis=0)
y2 = np.max(thresh41, axis=1)
z2 = np.max(thresh41, axis=2)

aspects41 = np.array([x2, y2, z2])

w=3
h=1
fig=plt.figure(figsize=(1, 3))
columns = 3
rows = 1
for i in range(1, columns*rows +1):
    img = aspects41[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

temp = aspects40[0]
query = aspects41[0]




ct = None
gc.collect()


#Find Keypoints
temp_kp, descriptors_temp = pysift.computeKeypointsAndDescriptors(temp)
query_kp, descriptors_query = pysift.computeKeypointsAndDescriptors(query)



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

print(good[0])

temp = to16bit(temp)
query = to16bit(query)

#Draw the lines
img3 = cv2.drawMatches(temp, temp_kp, query, query_kp, good, None, **draw_params)
plt.imshow(img3)
plt.axis('off')
plt.show()

new_x =cv2.warpPerspective(aspects41[0], M, (1000,1000)) 
ctreader.view(new_x)

