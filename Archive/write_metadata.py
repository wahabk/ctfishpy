from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
import csv

def write_metadata(n, input):
	fishnum = self.fishnums[n]
	fishpath = f'../../Data/HDD/low_res_clean/{fishnum}/'
	jsonpath = fishpath + 'metadata.json'
	jsonpath = './output/test.json' # this is the test

	with open(jsonpath) as f:
		metadata = json.load(f)

	for key in list(input.keys()):
		metadata[key] = input[key]
	
	with open(jsonpath, 'w') as o:
		json.dump(metadata, o)


meta = {
	'N'		  : None,
	'Skip'	   : None,
	'Age'		: None,
	'Genotype'   : None,
	'Strain'	 : None,
	'Name'	   	 : None,
	'VoxelSizeX' : None,
	'VoxelSizeY' : None,
	'VoxelSizeZ' : None
}

m = {'VoxelSizeZ' : 'kobe'}

CTreader = CTreader()
mastersheet = CTreader.mastersheet()
fishnum = 40
fish = mastersheet.loc[mastersheet['n'] == fishnum].to_dict()

print(fish['genotype'][0])




















