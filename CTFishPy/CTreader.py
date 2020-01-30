import CTFishPy.utility as utility
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

    def findrows(self, df, col, value):
        #Find all rows that have specified value in specified column
        #e.g. find all rows that have 12 in column 'age'
        return list(df.loc[df[col]==value].index.values)

    def trim(self, df, col, value):
        #Trim df to e.g. fish that are 12 years old
        index = findrows(df, col, value)
        trimmed = df.drop(set(df.index) - set(index))
        return trimmed

    def read(self, fish):
        pass

    def read_dirty(self, file_number = None, r = (1,100), scale = 40, color = False):
        path = '../../Data/HDD/uCT/low_res/'
        files = os.listdir(path)
        files = natsorted(files, alg=ns.IGNORECASE)
        files_df = pd.DataFrame(files)
        files_df.to_csv('../../Data/HDD/uCT/filenames_low_res.csv', index = False, header = False)
        if file_number == None:

            print(files)
            return

        file = files[file_number]
        paths = next(os.walk('../../Data/HDD/uCT/low_res/'+file+''))[1]
        
        # Find tif folder and if it doesnt exist read images in main folder
        tif = []
        for i in paths: 
            if i.startswith('EK'):
                tif.append(i)
        if tif:
            tifpath = path+file+'/'+tif[0]+'/'
        else:
            tifpath = path+file+'/'

        ct = []
        ct_color = []

        print('[FishPy] Reading uCT scan')
        for i in tqdm(range(*r)):
            x = cv2.imread(tifpath+file+'_'+(str(i).zfill(4))+'.tif')            
            height = int(x.shape[0] * scale / 100)
            width = int(x.shape[1] * scale / 100)
            x = cv2.resize(x, (width, height), interpolation = cv2.INTER_AREA)         
            x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            ct.append(x_gray)
            ct_color.append(x)
        ct = np.array(ct)
        ct_color = np.array(ct_color)

        # read xtekct
        if np.count_nonzero(ct) == 0:
            raise ValueError('Image is empty.')

        return ct, ct_color #ct: (slice, x, y), color: (slice, x, y, 3)

    def view(self, ct_array):
        fig, ax = plt.subplots(1, 1)
        tracker = utility.IndexTracker(ax, ct_array.T)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    def find_tubes(self, ct, minDistance = 200, 
        minRad = 50, maxRad = 150, thresh = [50, 100]):
        
        output = ct.copy()
        ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
        min_thresh, max_thresh = thresh
        ret, ct = cv2.threshold(ct, min_thresh, max_thresh, 
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        circles = cv2.HoughCircles(ct, cv2.HOUGH_GRADIENT, dp=1.5, 
        minDist = minDistance, minRadius = minRad, maxRadius = maxRad) #param1=50, param2=30,

        if circles is None:
            print('[FishPy] No circles found :(')
            return

        else:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

                # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 0, 255), 2)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            
            print('[FishPy] Tubes detected:', circles.shape[0])
            circle_dict =  {'labelled_img'  : output, 
                            'circles'       : circles}
            return circle_dict
            
    def crop(self, ct, circles, pad = 0):
        CTs = []
        for x, y, r in circles:
            c = []
            for slice_ in ct:
                rectx = x-r
                recty = y-r
                cropped_slice = slice_[recty-pad:(recty+2*r+pad), 
                    rectx-pad:(rectx+2*r+pad)]
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

'''
master = CTreader.mastersheet()
index = utility.findrows(master, 'age', 12)
oneyearolds = utility.trim(master, 'age', 12)
'''

#master['age'].value_counts()
