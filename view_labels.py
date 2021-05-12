import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path



mariele_sample 	= [40,242,256,259,421,423,459,463,530,589]
zac_sample 		= [40,242,256,257,421,423,443,461,527,582]
my_sample		= [78,200,218,240,330,337,341,364,385,462,464]
organ = 'Otoliths'
ctreader = ctfishpy.CTreader()


for n in my_sample:
	ct, stack_metadata = ctreader.read(n, align=True)
	datapath = ctreader.dataset_path
	label = ctreader.read_label(organ, n, align=False, is_amira=True)
	
	ctreader.view(ct, label=label)
