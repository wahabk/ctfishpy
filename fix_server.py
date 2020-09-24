import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from pathlib2 import Path
import json
random.seed(a = 111)

ctreader = ctfishpy.CTreader()
nums = ctreader.fish_nums

angles = {}
for fish in nums:
	ct, stack_metadata = ctreader.read(fish, align=True)
