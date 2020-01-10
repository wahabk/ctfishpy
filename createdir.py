from tools.utility import IndexTracker
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import json
import os
import os.path

def makedir(dirName):
	# Create target Directory if don't exist
	if not os.path.exists(dirName):
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created.")
	else:    
		print("Directory " , dirName ,  " already exists.", end="\r")


metadata = {
	'number':	None,
	'genotype':	None, 
	'age'	:	None,
	'x_size':	None,
	'y_size':	None,
	'z_size':	None
}


for i in range(40, 639):
	path = '../../Data/uCT/low_res_clean/' + str(i).zfill(3) + '/'
	tifpath = path + 'reconstructed_tifs/'
	jsonpath = path + 'metadata.json'
	makedir(path)
	makedir(tifpath)
	metadata['number'] = i
	with open(jsonpath, 'w') as o:
		json.dump(metadata, o)

metadata = {
'n':   None, 
'skip':   None, 
'age':   None, 
'genotype':   None, 
'strain':   None, 
'name':   None, 
're-uCT scan':   None,
'Comments':   None, 
'age(old)':   None, 
'Phantom':   None, 
'Scaling Value':   None, 
'Arb Value:   None'
}

