import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

ctreader = ctfishpy.CTreader()

labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'

#label = ctreader.read_labels(labelpath)
#ctreader.view(label[0])


ctreader = ctfishpy.CTreader()
ct, stack_metadata = ctreader.read(40, r = None)#(1400,1600))
label = ctreader.read_label(labelpath)

#ct = ct[:,:,:,np.newaxis] # add final axis to show datagen its grayscale
#label = label[:,:,:,np.newaxis]
ct = np.array([cv2.resize(slice_, (512,512)) for slice_ in ct])
label = np.array([cv2.resize(slice_, (512,512)) for slice_ in label])

ctreader.view(ct, label)
