from . GUI.mainviewer import mainViewer
from natsort import natsorted, ns
from qtpy.QtCore import QSettings
from pathlib2 import Path
from tqdm import tqdm
import pandas as pd
import numpy as np 
import json
import cv2
import os


class CTreader():
    def init(self):
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
        # func to read clean data

    def view(self, ct_array):
        mainViewer(ct_array)

    def read_dirty(self, file_number = None, r = None, 
        scale = 40):
        path = '../../Data/HDD/uCT/low_res/'
        
        # find all dirty scan folders and save as csv in directory
        files      = os.listdir(path)
        files      = natsorted(files, alg=ns.IGNORECASE) #sort according to names without leading zeroes
        files_df    = pd.DataFrame(files) #change to df to save as csv
        files_df.to_csv('../../Data/HDD/uCT/filenames_low_res.csv', index = False, header = False)
        fish_nums = []
        for f in files:
            nums = [int(i) for i in f.split('_') if i.isdigit()]
            if len(nums) == 2:
                start = nums[0]
                end = nums[1]+1
                nums = list(range(start, end))
            fish_nums.append(nums)
        self.fish_order_nums = fish_nums#[[files[i], fish_nums[i]] for i in range(0, len(files))]
        self.files = files

        # get rid of weird mac files
        for file in files:
            if file.endswith('DS_Store'): files.remove(file)

        # if no file number was provided to read then print files list
        if file_number == None: 
            print(files)
            return

        #f ind all dirs in scan folder
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

        tifpath = Path(tifpath)
        files = sorted(tifpath.iterdir())
        images = [str(f) for f in files if f.suffix == '.tif']

        ct = []
        print('[CTFishPy] Reading uCT scans')
        if r:
            for i in tqdm(range(*r)):
                slice_ = cv2.imread(images[i])     
                # use provided scale metric to downsize image
                height  = int(slice_.shape[0] * scale / 100)
                width   = int(slice_.shape[1] * scale / 100)
                slice_ = cv2.resize(slice_, (width, height), interpolation = cv2.INTER_AREA)     
                ct.append(slice_)
            ct = np.array(ct)

        else:
            for i in tqdm(images):
                slice_ = cv2.imread(i)     
                # use provided scale metric to downsize image
                height  = int(slice_.shape[0] * scale / 100)
                width   = int(slice_.shape[1] * scale / 100)
                slice_ = cv2.resize(slice_, (width, height), interpolation = cv2.INTER_AREA)     
                ct.append(slice_)
            ct = np.array(ct)

        # check if image is empty
        if np.count_nonzero(ct) == 0:
            raise ValueError('Image is empty.')

        # read xtekct
        path = Path(path) # change str path to pathlib format
        files = path.iterdir()
        xtekctpath = [str(f) for f in files if f.suffix == '.xtekct'][0]

        # check if xtekct exists
        if not Path(xtekctpath).is_file():
            raise Exception("[CTFishPy] XtekCT file not found. ")
        
        xtekct = QSettings(xtekctpath, QSettings.IniFormat)
        x_voxelsize = xtekct.value('XTekCT/VoxelSizeX')
        y_voxelsize = xtekct.value('XTekCT/VoxelSizeY')
        z_voxelsize = xtekct.value('XTekCT/VoxelSizeZ')

        metadata = {'path': str(path), 
                    'scale' : scale,
                    'x_voxel_size' : x_voxelsize,
                    'y_voxel_size' : y_voxelsize,
                    'z_voxel_size' : z_voxelsize}

        return ct, metadata # ct: (slice, x, y, 3)

    def find_tubes(self, ct, minDistance = 200, minRad = 0, maxRad = 150, 
        thresh = [50, 100], slice_to_detect = 0, dp = 1.3, pad = 0):
        # Find fish tubes
        # output = ct.copy() # copy stack to label later
        output = ct.copy()

        # Convert slice_to_detect to gray scale and threshold
        ct_slice_to_detect = cv2.cvtColor(ct[slice_to_detect], cv2.COLOR_BGR2GRAY)
        min_thresh, max_thresh = thresh
        ret, ct_slice_to_detect = cv2.threshold(ct_slice_to_detect, min_thresh, max_thresh, 
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # detect circles in designated slice
        circles = cv2.HoughCircles(ct_slice_to_detect, cv2.HOUGH_GRADIENT, dp=dp, 
        minDist = minDistance, minRadius = minRad, maxRadius = maxRad) #param1=50, param2=30,

        if circles is None: return
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
                             'circles'     : circles}
            return circle_dict
            
    def crop(self, ct, circles, scale = [40, 40]):
        # this is so ugly :(             scale = [from,to]
        # crop ct stack to circles provided in order
        
        # find scale factor of scale at which cropped and scale of current image
        scale_factor = scale[1]/scale[0]
        circles = [[int(x*scale_factor), int(y*scale_factor), int(r*scale_factor)] for x, y, r in circles]
        cropped_CTs = []

        ctx = ct.shape[2]
        cty = ct.shape[1]

        for x, y, r in circles:
            cropped_stack = []

            crop_length = 2*r
            rectx, recty = [], []
            rectx.append(x - r)
            rectx.append(rectx[0] + crop_length)
            recty.append(y - r)
            recty.append(recty[0] + crop_length)
            
            # if statements to shift crop inside ct window
            if rectx[0] < 0:
                shiftx = rectx[0]
                rectx[0] = 0
                rectx[1] = rectx[1] - shiftx

            if rectx[1] > ctx:
                shiftx = rectx[1] - ctx
                rectx[1] = ctx
                rectx[0] = rectx[0] + shiftx

            if recty[0] < 0:
                shifty = recty[0]
                recty[0] = 0
                recty[1] = recty[1] - shifty

            if recty[1] > cty:
                shifty = recty[1] - cty
                recty[1] = cty
                recty[0] = recty[0] + shifty

            for np_slice in ct:
                cropped_slice =  np_slice[
                    recty[0] : recty[1],
                    rectx[0] : rectx[1],
                             : ]
                    # x1  :  x2
                cropped_stack.append(cropped_slice)
            cropped_stack = np.array(cropped_stack, dtype = np.uint8)
            cropped_CTs.append(cropped_stack)
        return cropped_CTs

    def saveCrop(self, n, ordered_circles, metadata):
        fishnums = np.arange(40,639)
        number = fishnums[n]
        order = self.fish_order_nums[n]
        crop_data = {
            'n'                 : f'{order[0]}-{order[len(order)-1]}',
            'ordered_circles'   : ordered_circles.tolist(),
            'scale'             : metadata['scale'],
            'path'              : metadata['path']
        }

        jsonpath = metadata['path']+'/crop_data.json'
        with open(jsonpath, 'w') as o:
            json.dump(crop_data, o)
        backuppath = f'./output/Crops/{order[0]}-{order[len(order)-1]}_crop_data.json'
        with open(backuppath, 'w') as o:
            json.dump(crop_data, o)

    def readCrop(self, number):
        files = pd.read_csv('../../Data/HDD/uCT/filenames_low_res.csv', header = None)
        files = files.values.tolist()
        crop_path = '../../Data/HDD/uCT/low_res/'+files[number][0]+'/crop_data.json'
        with open(crop_path) as f:
            crop_data = json.load(f)
        return crop_data

    def write_metadata(self, n, input):
        '''
        metadata = {
            'N'    : None,
            'Skip'     : None,
            'Age'      : None,
            'Genotype'   : None,
            'Strain'     : None,
            'Name'     : None,
            'VoxelSizeX' : None,
            'VoxelSizeY' : None,
            'VoxelSizeZ' : None
        }
        '''
        #n = self.fishnums[n]
        fishpath = f'../../Data/HDD/uCT/low_res_clean/{str(n).zfill(3)}/'
        jsonpath = fishpath + 'metadata.json'

        with open(jsonpath) as f:
            metadata = json.load(f)

        for key in list(input.keys()):
            metadata[key] = input[key]

        with open(jsonpath, 'w') as o:
            json.dump(metadata, o)

    def write_clean(self, n, cropped_cts, metadata):
        order = self.fish_order_nums[n]
        if len(order) != len(cropped_cts): raise Exception('Not all/too many fish cropped')
        mastersheet = pd.read_csv('./uCT_mastersheet.csv')

        print(f'[CTFishPy] Writing cropped CT scans')
        for o in range(0, len(order)): # for each fish of number o
            path = f'../../Data/HDD/uCT/low_res_clean/{str(order[o]).zfill(3)}/'
            tifpath = path + 'reconstructed_tifs/'
            metapath = path + 'metadata.json'

            ct = cropped_cts[o]
            fish = mastersheet.loc[mastersheet['n'] == 100].to_dict()
            weird_fix = list(fish['age'].keys())[0]
            
            input_metadata = {
                'Skip'          : fish['skip'][weird_fix],
                'Age'           : fish['age'][weird_fix],
                'Genotype'      : fish['genotype'][weird_fix],
                'Strain'        : fish['strain'][weird_fix],
                'Name'          : fish['name'][weird_fix],
                'VoxelSizeX'    : metadata['x_voxel_size'],
                'VoxelSizeY'    : metadata['y_voxel_size'],
                'VoxelSizeZ'    : metadata['z_voxel_size'],
                'Comments'      : fish['name'][weird_fix],
                'Phantom'       : fish['name'][weird_fix],
                'Scaling Value' : fish['name'][weird_fix],
                'Arb Value'     : fish['name'][weird_fix]
            }

            self.write_metadata(order[o], input_metadata)

            i = 0
            for img in ct: # for each slice
                filename = tifpath+f'{str(order[o]).zfill(3)}_{str(i).zfill(4)}.png'
                if img.size == 0: raise Exception(f'cropped image is empty at fish: {o+1} slice: {i+1}')
                ret = cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                if not ret: raise Exception('image not saved, directory doesnt exist')
                i = i + 1
                print(f'[Fish {order[o]}, slice:{i}/{len(ct)}]', end="\r")

