import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()
	master = ctreader.mastersheet()


	fine = [401, 402, 403, 410, 423, 522, 542]
	bad_bois = [424, 429, 433, 434, 435, 465, 467, 468, 534, 543, 559]
	dirty = ['EK_423_430', 'EK_431_438', 'EK_464_470', 'EK_533_540', 'EK_541_548', 'EK_559_566']
	flipped = [223,262,295,501,543]

	print(ctreader.fish_nums)

	# for n in bad_bois:
	# 	scan, metadata = ctreader.read(n, align=True)
	# 	ctreader.view(scan) 

	for n in dirty:
		




