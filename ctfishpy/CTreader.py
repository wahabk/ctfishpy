from .GUI import mainviewer
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import cv2
import h5py

class CTreader():
    def __init__(self):
        self.master = pd.read_csv('./uCT_mastersheet.csv')
        self.fishnums = np.arange(40,639)

    def mastersheet(self):
        return self.master

    def trim(self, m, col, value):
        # Trim df to e.g. fish that are 12 years old
        # Find all rows that have specified value in specified column
        # e.g. find all rows that have 12 in column 'age'
        index = list(m.loc[m[col]==value].index.values)
        # delete ones not in index
        trimmed = m.drop(set(m.index) - set(index))
        return trimmed

    def list_numbers(self, m):
        return list(m.loc[:]['n'])

    def read(self, fish, r = None):
        fishpath = Path.home() / 'Data' / 'HDD' / 'uCT' / 'low_res_clean' / str(fish).zfill(3) 
        tifpath = fishpath / 'reconstructed_tifs'
        metadatapath = fishpath / 'metadata.json'

        metadatafile = metadatapath.open()
        stack_metadata = json.load(metadatafile)
        metadatafile.close()

        #images = list(tifpath.iterdir())
        images = [str(i) for i in tifpath.iterdir()]
        images.sort()

        ct = []
        print(f'[CTFishPy] Reading uCT scan. Fish: {fish}')
        if r:
            for i in tqdm(range(*r)):
                tiffslice = tiff.imread(images[i])
                #tiffslice = cv2.imread(images[i]) 
                ct.append(tiffslice)
            ct = np.array(ct)

        else:
            for i in tqdm(images):
                tiffslice = tiff.imread(i)
                #tiffslice = cv2.imread(i)
                ct.append(tiffslice)
            ct = np.array(ct)

        return ct, stack_metadata

    def view(self, ct_array, label = None, thresh = False):
        mainviewer.mainViewer(ct_array, label, thresh)

    def read_label(self, labelpath):
        print('[CTFishPy] Reading labels...')
        f = h5py.File(labelpath, 'r')
        label = np.array(f['t0']['channel0'])
        print('Labels ready.')
        return label

    def to8bit(self, img):
        '''
        Change img from 16bit to 8bit
        '''
        if img.dtype == 'uint16':
            new_img = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8) 
            return new_img
        else:
            print('Stack already 8 bit!')
            return stack

    def thresh_stack(self, stack, thresh_8):
        '''
        Threshold CT stack in 16 bits using numpy because it's faster
        provide threshold in 8bit since it's more intuitive then convert to 16
        '''

        thresh_16 = thresh_8 * (65535/255)

        thresholded = []
        for slice_ in stack:
            new_slice = (slice_ > thresh_16) * slice_
            thresholded.append(new_slice)
        return np.array(thresholded)

    def saveJSON(self, nparray, jsonpath):
        json.dump(nparray, codecs.open(jsonpath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

    def readJSON(self, jsonpath):
        obj_text = codecs.open(jsonpath, 'r', encoding='utf-8').read()
        obj = json.loads(obj_text)
        return np.array(obj)

    def get_max_projections(self, stack):
        '''
        return x, y, x which represent axial, saggital, and coronal max projections
        '''
        x = np.max(stack, axis=0)
        y = np.max(stack, axis=1)
        z = np.max(stack, axis=2)
        return [x, y, z]

    def resize(self, img, percent=100):
        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
