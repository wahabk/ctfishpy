import CTFishPy.utility as utility
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

    def read(self, fish):
        pass

    def read_dirty(self, file_number = None, r = (0,100), scale = 40, color = False):
        path = '../../Data/uCT/low_res/'
        files = os.listdir(path)
        with open('../../Data/uCT/filenames.csv','w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(files)
        if file_number == None:
            print(files)
            return

        file = files[file_number]
        reconstructed_tifs = path+file+'_reconstructed_tifs/'
        if not os.path.exists(reconstructed_tifs):
            reconstructed_tifs = path+file+'/'
        
        ct = []
        ct_color = []
        print('[FishPy] Reading uCT scan')
        for i in tqdm(range(*r)):
            x = cv2.imread(reconstructed_tifs+file+'_'+(str(i).zfill(4))+'.tif')            
            width = int(x.shape[1] * scale / 100)
            height = int(x.shape[0] * scale / 100)
            x = cv2.resize(x, (width, height), interpolation = cv2.INTER_AREA)         
            x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            ct.append(x_gray)
            ct_color.append(x)

        # read xtekct
        
        ct = np.array(ct)
        ct_color = (ct_color)
        return ct, ct_color

    def view(self, ct_array):
        fig, ax = plt.subplots(1, 1)
        tracker = utility.IndexTracker(ax, ct_array.T)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    def find_tubes(self, ct , minDistance = 150, minRad = 40):
        output = ct.copy()
        ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
        ret, ct = cv2.threshold(ct, 50, 100, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        circles = cv2.HoughCircles(ct, cv2.HOUGH_GRADIENT, dp=1.2, minDist = minDistance, minRadius = minRad) #param1=50, param2=30,


        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

                # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 0, 255), 2)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            return output, circles

        else:
            print('No circles found :(')

    def write_metadata(self, fish, metadata):
        pass

    def write_images(self, fish, ct):
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
