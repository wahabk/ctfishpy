import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import csv
from tqdm import tqdm
import CTFishPy.utility as utility

class CTreader():
    def init(self):
        pass

    def mastersheet(self):
        return pd.read_csv('./uCT_mastersheet.csv')

    def read(self, fish):
        pass

    def read_dirty(self, file_number = None, r = (0,100), scale_percent = 75):
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
        print('[FishPy] Reading uCT scan')
        for i in tqdm(range(*r)):
            x = cv2.imread(reconstructed_tifs+file+'_'+(str(i).zfill(4))+'.tif')            
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            width = int(x.shape[1] * scale_percent / 100)
            height = int(x.shape[0] * scale_percent / 100)
            x = cv2.resize(x, (width, height), interpolation = cv2.INTER_AREA)
            ct.append(x)
        # read xtekct
        ct = np.array(ct)
        return ct

    def view(self, ct_array):
        fig, ax = plt.subplots(1, 1)
        tracker = utility.IndexTracker(ax, ct_array.T)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()


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
