import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
import gc
random.seed(a = 111)

ctreader = ctfishpy.CTreader()


for fish in range(40,500)
	ct, metadata = ctreader.read(fish)

	projection = ctreader.get_max_projections(ct)[0]

	angle = ctreader.spin(projection)
	print(projection.shape)

	print(angle)