import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

ctreader = ctfishpy.CTreader()

labelpath = '../../Data/HDD/uCT/Labels/Otolith1/40.h5'

label = ctreader.read_label(labelpath)
#ctreader.view(label)
ct, stack_metadata = ctreader.read(40)
ctreader.view(ct[1370], label = label[1370])
#ctreader.view(ct, label = label)


