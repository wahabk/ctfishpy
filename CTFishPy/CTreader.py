from CTFishPy.GUI.view import view as guiview
from natsort import natsorted, ns
import matplotlib.pyplot as plt
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
        return pd.read_csv('./uCT_mastersheet.csv')
        #to count use master['age'].value_counts()

    def trim(self, df, col, value):#Trim df to e.g. fish that are 12 years old
        #Find all rows that have specified value in specified column
        #e.g. find all rows that have 12 in column 'age'
        index = list(df.loc[df[col]==value].index.values)
        #delete ones not in index
        trimmed = df.drop(set(df.index) - set(index))
        return trimmed

    def read(self, fish):
        pass
        #func to read clean data

    def read_dirty(self, file_number = None, r = (1,100), scale = 30, color = False):
        path = '../../Data/HDD/uCT/low_res/'
        
        #find all dirty scan folders and save as csv in directory
        files = os.listdir(path)
        files = natsorted(files, alg=ns.IGNORECASE) #sort according to names without leading zeroes
        files_df = pd.DataFrame(files) #change to df to save as csv
        files_df.to_csv('../../Data/HDD/uCT/filenames_low_res.csv', index = False, header = False)
        
        #if no file number was provided to read then print files list
        if file_number == None: 
            print(files)
            return

        #find all dirs in scan folder
        file = files[file_number]
        paths = next(os.walk('../../Data/HDD/uCT/low_res/'+file+''))[1]
        # Find tif folder and if it doesnt exist read images in main folder
        tif = []
        for i in paths: 
            if i.startswith('EK'):
                tif.append(i)
        if tif: tifpath = path+file+'/'+tif[0]+'/'
        else: tifpath = path+file+'/'


        ct = []
        ct_color = []
        print('[FishPy] Reading uCT scan')
        for i in tqdm(range(*r)):
            x = cv2.imread(tifpath+file+'_'+(str(i).zfill(4))+'.tif')         
            #use provided scale metric to downsize image
            height  = int(x.shape[0] * scale / 100)
            width   = int(x.shape[1] * scale / 100)
            x = cv2.resize(x, (width, height), interpolation = cv2.INTER_AREA)     
            #convert image to gray and save both color and gray stack
            x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            ct.append(x_gray)
            ct_color.append(x)
        ct = np.array(ct)
        ct_color = np.array(ct_color)

        # read xtekct

        #check if image is empty
        if np.count_nonzero(ct) == 0:
            raise ValueError('Image is empty.')
        return ct, ct_color #ct: (slice, x, y), color: (slice, x, y, 3)

    def view(self, ct_array):
        guiview(ct_array)

    def find_tubes(self, ct, minDistance = 200, minRad = 50, maxRad = 150, 
        thresh = [50, 100], slice_to_detect = 0, dp = 1.2):
        # Find fish tubes
        #output = ct.copy() # copy stack to label later
        output = []

        #Convert slice_to_detect to gray scale and threshold
        ct_slice_to_detect = cv2.cvtColor(ct[slice_to_detect], cv2.COLOR_BGR2GRAY)
        min_thresh, max_thresh = thresh
        ret, ct_slice_to_detect = cv2.threshold(ct_slice_to_detect, min_thresh, max_thresh, 
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #detect circles in designated slice
        circles = cv2.HoughCircles(ct_slice_to_detect, cv2.HOUGH_GRADIENT, dp=dp, 
        minDist = minDistance, minRadius = minRad, maxRadius = maxRad) #param1=50, param2=30,

        if circles is None:
            print('[FishPy] No circles found :(')
            return

        else:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int") # round up

            # loop over the (x, y) coordinates and radius of the circles
            for i in ct:
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(i, (x, y), r, (0, 0, 255), 2)
                    cv2.rectangle(i, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                output.append(i)
            output = np.array(output)
            
            print('[FishPy] Tubes detected:', circles.shape[0])
            circle_dict =  {'labelled_img'  : output[slice_to_detect],
                            'labelled_stack': output, 
                            'circles'      : circles}
            
            return circle_dict
            
    def crop(self, ct, circles, pad = 0):
        #this is so ugly :(
        #crop ct stack to circles provided in order
        CTs = []
        for x, y, r in circles:
            c = []
            for slice_ in ct:
                rectx = x - r
                recty = y - r
                cropped_slice =  slice_[
                    recty - pad : (recty + 2*r + pad), 
                    rectx - pad : (rectx + 2*r + pad)
                    ]#      x1  :  x2
                c.append(cropped_slice)
            c = np.array(c, dtype = np.uint8)
            CTs.append(c)
        return CTs

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


