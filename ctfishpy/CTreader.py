from . GUI.mainviewer import mainViewer
from pathlib2 import Path
from tqdm import tqdm
import pandas as pd
import numpy as np 
import json
import cv2
import os

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

    def read(self, fish):
        pass

        return ct, stack_metadata




    def view(self, ct_array):
        mainViewer(ct_array)
