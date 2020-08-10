import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
ctreader = ctfishpy.CTreader()
import gc
import json, codecs



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