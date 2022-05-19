from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.tubeDetector import detectTubes
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
import csv
import os
from natsort import natsorted, ns

path = '../../Data/HDD/uCT/low_res/'

#find all dirty scan folders and save as csv in directory
files       = os.listdir(path)
files       = natsorted(files, alg=ns.IGNORECASE)
fish_nums = []
for f in files:
	fish_nums.append([i for i in f.split('_') if i.isdigit()])
fish_order_nums = [[files[i], fish_nums[i]] for i in range(0, len(files))]



