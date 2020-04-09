from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import ctfishpy

def DataGenie(batch_size, aug_dict, image_color_mode = "grayscale",
                mask_color_mode = "grayscale", target_size = (256,256)):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)


    fish_numbers = [40, 41, 42]
    ctreader = CTreader()
    fish = ctreader.read(fish_numbers[0])

    i = 0
    while(True):
        
        batch_images = []
        batch_masks = []

        if number_images == batch_size:
            break

        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            yield (img,mask)



data_gen_args = dict(rotation_range=180,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.2,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

datagen = ImageDataGenerator(**data_gen_args)

#g = dataGenie(batch_size = 2, data_gen_args, save_to_dir = None)
labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'
ctreader = ctfishpy.CTreader()

ct, stack_metadata = ctreader.read(40, r = (1400,1600))
label = ctreader.read_label(labelpath)

ct = ct[:,:,:,np.newaxis] # add final axis to show datagen its grayscale
d = datagen.flow(ct, 
    batch_size = 100,
    save_to_dir = 'output/Keras/',
    save_prefix = 'dataGenie',
    seed = 420
    )

count = 0
for x_batch in d:
    print(x_batch.shape)
    print(x_batch.dtype)
    break



