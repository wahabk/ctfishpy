import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	master = ctreader.mastersheet()
	datapath = 'output/otolithddatacol11.csv'

	# strains = ['col11a2']
	# master = master[master['strain'].isin(strains)]
	# genotypes = ['wt']
	# master = master[master['genotype'].isin(genotypes)]
	# ages = ['12']
	# master = master[master['age'].isin(ages)]
	# print(master)
	# print(list(master['n']))
	# print(list(master['age']))

	homs = [421, 443, 582, 583, 584, 585, 586, 587, 588, 589]
	het7month = [459, 460, 461]
	het18month = [256, 257, 258, 259]
	het28month = [462, 463, 464]
	wt12month = [68, 69, 70, 71, 72, 73, 74, 102, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 247, 431, 432, 433, 434, 440, 441, 442, 574, 575, 576]

	for n in het18month+het28month+het7month:
		ct, metadata = ctreader.read(n, align=True)
		center = ctreader.manual_centers[str(n)]

		otolith = ctreader.crop3d(ct, [200,200,200], center=center)
		ctreader.view(otolith)

		

