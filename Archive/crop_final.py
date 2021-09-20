import ctfishpy
from ctfishpy.viewer.circle_order_labeller import circle_order_labeller
from ctfishpy.viewer.tubeDetector import detectTubes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
from pathlib2 import Path

CTreader = ctfishpy.controller.CTreader()
lump = ctfishpy.controller.Lumpfish()


for i in range(62,63):
	print(i)
	ct, stack_metadata = lump.read_dirty(i, r = None, scale = 40)
	circle_dict = detectTubes(ct)
	CTreader.view(ct)
	ordered_circles, numbered = circle_order_labeller(
		circle_dict['labelled_stack'], circle_dict['circles'])
	CTreader.view(numbered)
	lump.saveCrop(n = i, 
		ordered_circles = ordered_circles, 
		metadata = stack_metadata)
