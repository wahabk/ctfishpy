import CTFishPy.utility as utility
from CTFishPy.CTreader import CTreader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
pd.set_option('display.max_rows', None)

#master['age'].value_counts()

CTreader = CTreader()

'''
master = CTreader.mastersheet()

index = utility.findrows(master, 'age', 12)

oneyearolds = utility.trim(master, 'age', 12)

print(oneyearolds)
'''


ct = CTreader.read_dirty(0)
CTreader.view(ct)


