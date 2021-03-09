import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from pathlib2 import Path



mariele_sample 	= [40,242,256,259,421,423,459,463,530,589]
zac_sample 		= [40,242,256,257,421,423,443,461,527,582]
organ = 'Otoliths_Mariele'
ctreader = ctfishpy.CTreader()


for n in mariele_sample:
	ct, stack_metadata = ctreader.read(n, align=True)
	datapath = ctreader.dataset_path
	label = ctreader.read_label(organ, n, align=False, is_amira=True)
	
	ctreader.view(ct, label=label)
