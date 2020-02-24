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
	ct, stack_metadata = CTreader.read_dirty(i, r = (900,1100), scale = 40)

	crop_data = CTreader.readCrop(i)
	scale = [crop_data['scale'], stack_metadata['scale']]
	cropped_ordered_cts = CTreader.crop(ct, crop_data['ordered_circles'], scale = scale)

	CTreader.write_clean(i, cropped_ordered_cts, stack_metadata)
