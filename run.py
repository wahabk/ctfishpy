import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

ctreader = ctfishpy.CTreader()

labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'

#label = ctreader.read_labels(labelpath)
#ctreader.view(label[0])

m = ctreader.mastersheet()

print(m)

