import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
random.seed(a = 111)

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
    plt.close()

ctreader = ctfishpy.CTreader()
for i in range(350, 353):
	ct, stack_metadata = ctreader.read(213)
	aspects = get_max_projections(ct)
	
	plot_list_of_3_images(aspects)

	#ctreader.view(ct)
