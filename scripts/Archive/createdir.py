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
		print("Created.", end="\r")
	else:    
		print("Already exists.", end="\r")

metadata = {
	'Number'		: None,
    'Skip'      	: None,
    'Age'       	: None,
    'Genotype'  	: None,
    'Strain'    	: None,
    'Name'      	: None,
    'VoxelSizeX'	: None,
    'VoxelSizeY'	: None,
    'VoxelSizeZ'	: None,
    'Comments'  	: None,
    'Phantom'   	: None,
    'Scaling Value' : None,
    'Arb Value' 	: None,
}

for i in range(40, 639):
	path = '../../Data/HDD/uCT/low_res_clean/' + str(i).zfill(3) + '/'
	tifpath = path + 'reconstructed_tifs/'
	jsonpath = path + 'metadata.json'
	wrongpath = path + f'{i}_metadata.json'
	os.remove(wrongpath)
	makedir(path)
	makedir(tifpath)
	metadata['Number'] = i
	metadata_json = json.dumps(metadata, indent=4)
	with open(jsonpath, 'w') as o:
		json.dump(metadata, o)

