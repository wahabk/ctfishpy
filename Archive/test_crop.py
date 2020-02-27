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

ct, stack_metadata = CTreader.read_dirty(0, r = (900,1100), scale = 40)

circle_dict = detectTubes(ct)

ordered_circles, numbered = circle_order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])

cropped_ordered_cts = CTreader.crop(ct, ordered_circles)

for c in cropped_ordered_cts: CTreader.view(c)
