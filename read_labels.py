import ctfishpy
from ctfishpy.read_amira import read_amira
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path
import h5py

ctreader = ctfishpy.CTreader()
labelpath = Path('../../Data/HDD/uCT/Labels/Otolith1/040.h5')

label = ctreader.read_labels(labelpath)
ctreader.view(label)

