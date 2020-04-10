import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

ctreader = ctfishpy.CTreader()

labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'

label = ctreader.read_label(labelpath)
#ctreader.view(label)
ct, stack_metadata = ctreader.read(40)

ctreader.view(ct, label = label)



