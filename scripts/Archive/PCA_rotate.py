import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import json, codecs
from math import atan2, cos, sin, sqrt, pi, degrees

ctreader = ctfishpy.CTreader()

# https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html
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


def PCA(projection, threshold = 150, scale = 75):
    color = ctreader.resize(projection, scale)

    cv2.imshow('raw projection', color)
    cv2.waitKey()
    cv2.destroyAllWindows()

    ret, thresh = cv2.threshold(color, 200, 255, cv2.THRESH_BINARY) #| cv2.THRESH_OTSU)
    bw = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
    cv2.imshow('thresholded then blurred', bw)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    blur_kernel = 43
    blur = cv2.GaussianBlur(bw, (blur_kernel, blur_kernel), 0)
    cv2.imshow('thresholded then blurred', blur)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #ret, blur = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY) #| cv2.THRESH_OTSU)
    cv2.imshow('thresholded then blurred', blur)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # cv2.imwrite('output/pcastep1.png', img)
    # cv2.imwrite('output/pcastep2.png', bw)
    # cv2.imwrite('output/pcastep3.png', blur)
    # cv2.imwrite('output/pcastep4.png', color)
    ## [contours]
    # Find all the contours in the thresholded image

    contours, _ = cv2.findContours(blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    i = areas.index(max(areas))
    # Draw each contour only for visualisation purposes
    cv2.drawContours(color, contours, i, (0, 0, 255), 2)
    # Find the orientation of each shape
    angle = getOrientation(contours[i], color)

    rotated = rotate_image(color, angle+180)

    ## [contours]
    cv2.imshow('output', rotated)
    cv2.imwrite('output/pcarotated.png', rotated)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return angle

if __name__ == "__main__":   
    ctreader = ctfishpy.CTreader()
    lumpfish = ctfishpy.Lumpfish()
    for i in [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]:
        projection = cv2.imread(f'Data/projections/x/{i}.png')
        #Add gaussian blur to combine otoliths
        angle = PCA(projection, threshold=150)
        print(angle)
        angleDict = {'angle': angle}
        lumpfish.append_metadata(i, angleDict)