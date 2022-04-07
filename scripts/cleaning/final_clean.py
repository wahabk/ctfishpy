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

	dirty_path = Path("/home/wahab/Data/Local/uCT/low_res")

	fine = [401, 402, 403, 410, 423, 522, 542]
	bad_bois = [424, 429, 433, 434, 435, 465, 467, 468, 534, 543, 559]
	dirty = ['EK_423_430', 'EK_431_438', 'EK_464_470', 'EK_533_540', 'EK_541_548', 'EK_559_566']
	flipped = [223,262,295,501,543]

	# for n in bad_bois:
	# 	scan, metadata = ctreader.read(n, align=True)
	# 	ctreader.view(scan) 

	for n in dirty:
		path = dirty_path / n
		scan, group_metadata = lump.read_dirty(path, r=(1000,1010), scale=100)
		print(group_metadata)

		scale_40 = lump.rescale(scan, 40)

		# ctreader.view(scale_40)

		# detect tubes
		circle_dict = lump.detectTubes(scale_40)

		print(circle_dict['circles'])

		ordered = lump.labelOrder(circle_dict)

		exit()



		#label order


		#crop


		#spin

	# save temp metadata with shape as practice

		




