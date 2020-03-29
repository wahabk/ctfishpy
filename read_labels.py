import ctfishpy
from ctfishpy.read_amira import read_amira
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path
import tifffile as tiff
import h5py
import ahds


labelspath = Path('../../Data/HDD/uCT/low_res_clean/040/40_lagenal_otoliths_labels.am')


#labels = tiff.imread(str(labelspath))
#print(labels.shape)
'''
f = h5py.File(labelspath, 'r')
print(list(f.keys()))
print(list(f['__DATA_TYPES__']))

image = np.array(f['t0']['channel0'])
'''
ctreader = ctfishpy.CTreader()

path = '../../Data/HDD/uCT/low_res_clean/040/40_lagenal_otoliths_labels.am'

labels = ahds.AmiraFile(path, load_streams = False)
label = labels.read()
print(labels.shape)



#ctreader.view(image)

