from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.tubeDetector import detectTubes
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
import csv

CTreader = CTreader()

for i in range(0,64):
	ct, stack_metadata = CTreader.read_dirty(i, r = None, scale = 40)
	circle_dict = detectTubes(ct)
	CTreader.view(ct)
	ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
	CTreader.view(numbered)
	CTreader.saveCrop(number = i, 
		ordered_circles = ordered_circles, 
		metadata = stack_metadata)

