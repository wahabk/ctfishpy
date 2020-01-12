import csv
import os
import numpy as np
import pandas as pd

path = '../../Data/uCT/low_res/'

files = os.listdir(path)

with open('../../Data/uCT/low_res/filenames.csv','w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(files)
