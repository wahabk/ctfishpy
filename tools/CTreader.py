import numpy as np 
import matplotlib.pyplt as pypl
import cv2
import os
import csv

class CTreader():
    def init(self):
        pass

        self.master = 

    def encode(self):
        pass

    def read(self, int = fish):
        self.label
        self.genotype
        self.age
        pass

    def read_dirty(self, int = fish):
        pass
        'xtekct :('

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

'../../Data/uCT/low_res_clean/208/'

try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")


stock_directory = 
{
  '../../Data/uCT/low_res_clean/'+fish+'': {
        'metadata': 'meta.json',
        "reconstructed_tifs": {
            "item1": None
        }
    }
}






#need a function to read amira labels



