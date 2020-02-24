from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.mainwindowcircle import detectTubes
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


c = CTreader()

for i in range(0,64):
	ct, stack_metadata = c.read_dirty(i, r = (0,5))

