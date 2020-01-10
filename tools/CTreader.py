import numpy as np 
import pandas as pd
import matplotlib.pyplot as pypl
import cv2
import os

class CTreader():
    def init(self):
        pass

    def mastersheet(self):
        return pd.read_csv('./uCT_mastersheet.csv')

    def read(self, fish):
        pass

    def read_dirty(self, fish):
        pass
        # read xtekct


'''
class Fish():
    def init(self, ct, metadata):
        pass
        self.ct = ct
        self.number     = metadata['number']
        self.genotype   = metadata['genotype']
        self.age        = metadata['age']
        self.x_size     = metadata['x_size']
        self.y_size     = metadata['y_size']
        self.z_size     = metadata['z_size']

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

'''
