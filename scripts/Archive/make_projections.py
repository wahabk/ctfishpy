import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
from pathlib2 import Path
random.seed(a = 111)

ctreader = ctfishpy.CTreader()

dataset = Path('/home/wahab/Data/HDD/uCT/low_res_clean/')
nums = [int(path.stem) for path in dataset.iterdir() if path.is_dir()]
nums.sort()
projections = Path('Data/projections/x/')
made = [int(path.stem) for path in projections.iterdir() if path.is_file() and path.suffix == '.png']
made.sort()
nums = [x for x in nums if x not in made]
nums.sort()
print(nums)


for fish in nums:
	ct, stack_metadata = ctreader.read(fish, align=True)
	z, x, y = ctreader.make_max_projections(ct)
	cv2.imwrite(f'../../Data/HDD/uCT/projections/x/x_{fish}.png', x)
	cv2.imwrite(f'../../Data/HDD/uCT/projections/y/y_{fish}.png', y)
	cv2.imwrite(f'../../Data/HDD/uCT/projections/z/z_{fish}.png', z)


