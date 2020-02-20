from . GUI.view import view as guiview
from natsort import natsorted, ns
from qtpy.QtCore import QSettings
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np 
import csv
import cv2
import os

class CTreader():
    def init(self):
        pass

    def mastersheet(self):
        self.mastersheet = pd.read_csv('./uCT_mastersheet.csv')
        return self.mastersheet
        #to count use master['age'].value_counts()

    def trim(self, col, value):#Trim df to e.g. fish that are 12 years old
        #Find all rows that have specified value in specified column
        #e.g. find all rows that have 12 in column 'age'
        m = self.mastersheet
        index = list(m.loc[m[col]==value].index.values)
        #delete ones not in index
        trimmed = m.drop(set(m.index) - set(index))
        return trimmed

    def read(self, fish):
        pass
        #func to read clean data

    def read_dirty(self, file_number = None, r = (1,100), 
        scale = 40, color = False):
        path = '../../Data/HDD/uCT/low_res/'
        
        #find all dirty scan folders and save as csv in directory
        files       = os.listdir(path)
        files       = natsorted(files, alg=ns.IGNORECASE) #sort according to names without leading zeroes
        files_df    = pd.DataFrame(files) #change to df to save as csv
        files_df.to_csv('../../Data/HDD/uCT/filenames_low_res.csv', index = False, header = False)
        
        #get rid of weird mac files
        for file in files:
            if file.endswith('DS_Store'): files.remove(file)

        #if no file number was provided to read then print files list
        if file_number == None: 
            print(files)
            return

        #find all dirs in scan folder
        file = files[file_number]
        for path, dirs, files in os.walk('../../Data/HDD/uCT/low_res/'+file+''):
            dirs = sorted(dirs)
            break

        # Find tif folder and if it doesnt exist read images in main folder
        tif = []
        for i in dirs: 
            if i.startswith('EK'):
                tif.append(i)
        if tif: tifpath = path+'/'+tif[0]+'/'
        else: tifpath = path+'/'

        ct = []
        ct_color = []
        print('[FishPy] Reading uCT scan')
        for i in tqdm(range(*r)):
            #if i == 891: continue # skip this file
            x = cv2.imread(tifpath+file+'_'+(str(i).zfill(4))+'.tif')         
            # use provided scale metric to downsize image
            height  = int(x.shape[0] * scale / 100)
            width   = int(x.shape[1] * scale / 100)
            x = cv2.resize(x, (width, height), interpolation = cv2.INTER_AREA)     
            # convert image to gray and save both color and gray stack
            # x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            # ct.append(x_gray)
            ct_color.append(x)
        # ct = np.array(ct)
        ct_color = np.array(ct_color)

        # check if image is empty
        if np.count_nonzero(ct_color) == 0:
            raise ValueError('Image is empty.')

        # read xtekct
        xtekctpath = path+'/'+file+'.xtekct'

        if not Path(xtekctpath).is_file():
            print ("[CTFishPy] XtekCT file not found. ")
        
        xtekct = QSettings(xtekctpath, QSettings.IniFormat)
        x_voxelsize = xtekct.value('XTekCT/VoxelSizeX')
        y_voxelsize = xtekct.value('XTekCT/VoxelSizeY')
        z_voxelsize = xtekct.value('XTekCT/VoxelSizeZ')

        metadata = {'path': path, 
                    'scale' : scale,
                    'x_voxel_size' : x_voxelsize,
                    'y_voxel_size' : y_voxelsize,
                    'z_voxel_size' : z_voxelsize}

        return ct_color, metadata #ct: (slice, x, y), color: (slice, x, y, 3)

    def view(self, ct_array):
        guiview(ct_array)

    def find_tubes(self, ct, minDistance = 200, minRad = 0, maxRad = 150, 
        thresh = [50, 100], slice_to_detect = 0, dp = 1.3, pad = 0):
        # Find fish tubes
        #output = ct.copy() # copy stack to label later
        output = ct.copy()

        #Convert slice_to_detect to gray scale and threshold
        ct_slice_to_detect = cv2.cvtColor(ct[slice_to_detect], cv2.COLOR_BGR2GRAY)
        min_thresh, max_thresh = thresh
        ret, ct_slice_to_detect = cv2.threshold(ct_slice_to_detect, min_thresh, max_thresh, 
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #detect circles in designated slice
        circles = cv2.HoughCircles(ct_slice_to_detect, cv2.HOUGH_GRADIENT, dp=dp, 
        minDist = minDistance, minRadius = minRad, maxRadius = maxRad) #param1=50, param2=30,

        if circles is None:
            return

        else:
            # add pad value to radii

            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int") # round up
            circles[:,2] = circles[:,2] + pad

            # loop over the (x, y) coordinates and radius of the circles
            for i in output:
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(i, (x, y), r, (0, 0, 255), 2)
                    cv2.rectangle(i, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            

            circle_dict  =  {'labelled_img'  : output[slice_to_detect],
                             'labelled_stack': output, 
                             'circles'       : circles}
            return circle_dict
            
    def crop(self, ct, circles, scale = [40, 40]):
        # this is so ugly :(             scale = [from,to]
        # crop ct stack to circles provided in order
        
        # find scale factor of scale at which cropped and scale of current image
        scale_factor = scale[1]/scale[0]
        circles = [[int(x*scale_factor), int(y*scale_factor), int(r*scale_factor)] for x, y, r in circles]
        cropped_CTs = []
        
        for x, y, r in circles:
            cropped_stack = []
            for slice_ in ct:
                rectx = x - r
                recty = y - r
                cropped_slice =  slice_[ 
                    recty : (recty + 2*r),
                    rectx : (rectx + 2*r),
                          : ]
                    # x1  :  x2
                cropped_stack.append(cropped_slice)
            cropped_stack = np.array(cropped_stack, dtype = np.uint8)
            cropped_CTs.append(cropped_stack)
        return cropped_CTs

    def saveCrop(self, ordered_circles, metadata):
        crop_data = {
            'ordered_circles'   : ordered_circles.tolist(),
            'scale'             : metadata['scale'],
            'path'              : metadata['path']
        }
        jsonpath = metadata['path']+'/crop_data.json'
        with open(jsonpath, 'w') as o:
            json.dump(crop_data, o)

    def readCrop(self, number):
        files = pd.read_csv('../../Data/HDD/uCT/filenames_low_res.csv', header = None)
        files = files.values.tolist()
        crop_path = '../../Data/HDD/uCT/low_res/'+files[number][0]+'/crop_data.json'
        with open(crop_path) as f:
            crop_data = json.load(f)
        return crop_data

    def write_metadata(self):
        pass

    def write_images(self):
        pass

'''
class Fish():
    def init(self, ct, metadata):
        pass
        self.ct = ct
        self.number  = metadata['number']
        self.genotype   = metadata['genotype']
        self.age        = metadata['age']
        self.x_size  = metadata['x_size']
        self.y_size  = metadata['y_size']
        self.z_size  = metadata['z_size']

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


