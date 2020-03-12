from . GUI.mainviewer import mainViewer
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import cv2
        
class CTreader():
    def __init__(self):
        self.mastersheet = pd.read_csv('./uCT_mastersheet.csv')
        self.fishnums = np.arange(40,639)

    def mastersheet(self):
        return pd.read_csv('./uCT_mastersheet.csv')
        #to count use master['age'].value_counts()

    def trim(self, col, value):
        # Trim df to e.g. fish that are 12 years old
        # Find all rows that have specified value in specified column
        # e.g. find all rows that have 12 in column 'age'
        m = self.mastersheet
        index = list(m.loc[m[col]==value].index.values)
        # delete ones not in index
        trimmed = m.drop(set(m.index) - set(index))
        return trimmed

    def read(self, fish, r = None):
        fishpath = Path.home() / 'Data' / 'HDD' / 'uCT' / 'low_res_clean' / str(fish).zfill(3) 
        tifpath = fishpath / 'reconstructed_tifs'
        metadatapath = fishpath / 'metadata.json'

        metadatafile = metadatapath.open()
        stack_metadata = json.load(metadatafile)
        metadatafile.close()

        images = list(tifpath.iterdir())
        images = [str(i) for i in tifpath.iterdir()]

        ct = []
        print('[CTFishPy] Reading uCT scans')
        if r:
            for i in tqdm(range(*r)):
                #tiffslice = tiff.imread(images[i])
                tiffslice = cv2.imread(images[i]) 
                ct.append(tiffslice)
            ct = np.array(ct)

        else:
            for i in tqdm(images):
                #tiffslice = tiff.imread(i)
                tiffslice = cv2.imread(i)
                ct.append(tiffslice)
            ct = np.array(ct)

        return ct, stack_metadata

    def view(self, ct_array):
        mainViewer(ct_array)
