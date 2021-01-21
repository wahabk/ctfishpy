import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from scipy import ndimage
from pathlib2 import Path
import json
random.seed(a = 111)

ctreader = ctfishpy.CTreader()
lumpfish = ctfishpy.Lumpfish()

nums = ctreader.fish_nums
nums.sort()
anglePath = ctreader.anglePath
with open(anglePath, 'r') as fp:
	angles = json.load(fp)

made = [int(key) for key in angles]
made.sort()
nums = [x for x in nums if x not in made]
nums.sort()
print(nums)

for fish in nums:
	metadata = ctreader.read_metadata(fish)
	# z, x, y = ctreader.read_max_projections(stack)
	# angle = ctreader.spin(z)
	angle = metadata['angle']

	lumpfish.write_metadata(fish, metadata)
	angles[str(fish)] = angle

with open(anglePath, 'w') as fp:
	json.dump(angles, fp)