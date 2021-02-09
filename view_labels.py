import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path



mariele_sample 	= [40,242,256,259,421,423,459,463,530,589]
zac_sample 		= [40,242,256,257,421,423,443,461,527,582]
organ = 'Otoliths_Zac'
ctreader = ctfishpy.CTreader()


for n in zac_sample:
	ct, stack_metadata = ctreader.read(n, align=True)
	datapath = ctreader.dataset_path
	path = datapath / f'Labels/Organs/{organ}/{n}.am'
	label_dict = read_amira(path)
	label = label_dict['data'][-1]['data'].T
	
	ctreader.view(ct, label=label)
