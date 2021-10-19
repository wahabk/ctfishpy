import ctfishpy
import matplotlib.pyplot as plt
import numpy as np
import json


if __name__ == '__main__':
	ctreader = ctfishpy.CTreader()
	segs = 'Otoliths_unet2d'
	nclasses = 3
	datapath = 'output/otolithddatacol11.csv'

	