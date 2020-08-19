import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import json, codecs
from math import atan2, cos, sin, sqrt, pi, degrees

ctreader = ctfishpy.CTreader()

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

    return degrees(angle)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

areas = []

def PCA(stack, threshold = 150, scale = 75):
    projections = ctreader.get_max_projections(stack)
    img = ctreader.resize(projections[0], scale)
    img = ctreader.to8bit(img)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    ret, bw = cv2.threshold(img, 50, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('', bw)
    cv2.waitKey()

    ## [contours]
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        areas.append(area)
        # Ignore contours that are too small or too large
        min_size = 10000
        if area < min_size:
            continue

        # Draw each contour only for visualisation purposes
        cv2.drawContours(color, contours, i, (0, 0, 255), 2)
        # Find the orientation of each shape
        angle = getOrientation(c, color)

        rotated = rotate_image(color, angle+90)

    ## [contours]

    cv2.imshow('output', rotated)
    cv2.waitKey()

ct, stack_metadata = ctreader.read(56)
angle = PCA(ct)
