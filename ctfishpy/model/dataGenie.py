from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from ..controller import CTreader
#from ..controller import cc
import gc
import cv2
import json
import matplotlib.pyplot as plt

def adjustData(img,mask,flag_multi_class=False,num_class=None):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 65535
        # mask = mask /255
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
    return (img,mask)

def fixFormat(batch, label = False):
    # change format of image batches to make viewable with ctreader
    if not label: return np.squeeze(batch.astype('uint16'), axis = 3)
    if label: return np.squeeze(batch.astype('uint8'), axis = 3)

def dataGenie(batch_size, data_gen_args, fish_nums = None):
    imagegen = ImageDataGenerator(**data_gen_args)
    maskgen = ImageDataGenerator(**data_gen_args)
    ctreader = CTreader()
    cc_centres_path = ctreader.dataset_path / 'cc_centres.json'
    with open(cc_centres_path, 'r') as fp:
        centres = json.load(fp)
    while True:
        ct_list, label_list = [], []
        for num in fish_nums:
            templatePath = '../../Data/HDD/uCT/Labels/CC/otolith_template_10.hdf5'
            labelpath = ctreader.dataset_path / 'Labels/Organs/'
            center = centres[str(num)]


            # take out cc for now
            # template = ctreader.read_label(templatePath, manual=False)
            # projections = ctreader.get_max_projections(num)
            # center, error = cc(num, template, thresh=200, roiSize=50)
            
            z_center = center[0] # Find center of cc result and only read roi from slices
            roiZ = 100

            ct, stack_metadata = ctreader.read(num, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)
            label = ctreader.read_label1(f'../../Data/HDD/uCT/Labels/Organs//Otolith1/{num}.h5', n=num,  align=True, manual=True)
            
            label = ctreader.crop_around_center3d(label, center = center, roiSize=128, roiZ=roiZ)
            center[0] = int(roiZ/2) # Change center to 0 because only read necessary slices but cant do that with labels since hdf5
            ct = ctreader.crop_around_center3d(ct, center = center, roiSize=128, roiZ=roiZ)
            ct_list.append(ct)
            label_list.append(label)
            ct, label = None, None
        
        ct_list = np.vstack(ct_list)
        label_list = np.vstack(label_list)
        ct_list = [cv2.resize(s, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for s in ct_list]
        label_list = [cv2.resize(s, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for s in label_list]

        ct_list = np.array(ct_list, dtype='float32')
        label_list = np.array(label_list, dtype='float32')
        
        # import pdb; pdb.set_trace()
        ct_list      = ct_list[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
        label_list   = label_list[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
        print('[dataGenie] Initialising image and mask generators')

        image_generator = imagegen.flow(ct_list,
            batch_size = batch_size,
            #save_to_dir = 'output/Keras/',
            save_prefix = 'dataGenie',
            seed = 42069,
            shuffle=True
            )
        mask_generator = maskgen.flow(label_list, 
            batch_size = batch_size,
            #save_to_dir = 'output/Keras/',
            save_prefix = 'dataGenie',
            seed = 42069,
            shuffle=True
            )
        print('Ready.')

        datagen = zip(image_generator, mask_generator)
        for x_batch, y_batch in datagen:
            #x_batch,y_batch = adjustData(x_batch,y_batch)
            x_batch = x_batch/65535
            # print(x_batch[0].shape, x_batch[0].dtype, np.amax(x_batch[0]))
            # print(y_batch[0].shape, y_batch[0].dtype, np.amax(y_batch[0]))
            yield (x_batch, y_batch)

def testGenie(num, batch_size=16):
    templatePath = '../../Data/HDD/uCT/Labels/CC/otolith_template_10.hdf5'
    ctreader = CTreader()
    template = ctreader.read_label1(templatePath, manual=False)
    # center, error = cc(num, template, thresh=200, roiSize=50)
    cc_centres_path = ctreader.dataset_path / 'cc_centres.json'
    with open(cc_centres_path, 'r') as fp:
        centres = json.load(fp)
    center = centres[str(num)]
    z_center = center[0] # Find center of cc result and only read roi from slices
    roiZ = 100
    ct, stack_metadata = ctreader.read(num, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)#(1400,1600))
    center[0] = int(roiZ/2)
    ct = ctreader.crop_around_center3d(ct, center = center, roiSize=128, roiZ=roiZ)
    # ctreader.view(ct)
    
    ct = [cv2.resize(s, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for s in ct]
    ct = np.array([_slice / 65535 for _slice in ct], dtype='float32') # Normalise 16 bit slices
    ct = ct[:,:,:,np.newaxis] # add final axis to show datagens its grayscale

    #print(ct.shape, np.amax(ct))
    
    # for i in range(0, ct.shape[0], batch_size):
    #     yield ct[i:i+batch_size]
    return ct



if __name__ == '__main__':
    data_gen_args = dict(rotation_range=180,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.2,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        vertical_flip = True,
                        fill_mode='constant',
                        cval = 0)
    batch_size = 100

    sample = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]
    # change label path to read labels directly

    datagenie = DataGenie(  batch_size = batch_size,
                            data_gen_args = data_gen_args,
                            fish_nums = sample)

    ctreader = ctfishpy.CTreader()

    for x_batch, y_batch in datagenie:
        #print(x_batch.shape)
        #print(y_batch.shape)
        x_batch = fixFormat(x_batch)  # remove last weird axis
        y_batch = fixFormat(y_batch, label = True)  # remove last weird axis

        ctreader.view(x_batch, label = y_batch)

        #break
