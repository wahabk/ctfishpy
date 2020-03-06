from ctfishpy.GUI.circle_order_labeller import circle_order_labeller
from ctfishpy.GUI.mainwindowcircle import detectTubes
from ctfishpy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


c = CTreader()

for i in range(0,64):
	ct, stack_metadata = c.read_dirty(i, r = (0,5))

