from __future__ import print_function
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



data_gen_args = dict(theta = 180, #rotation range
                    zx = 100,
                    zy = 100,
                    flip_horizontal = True,
                    flip_vertical = True)

datagen = ImageDataGenerator(data_gen_args)

#g = dataGenie(batch_size = 2, data_gen_args, save_to_dir = None)
labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'
fish_numbers = [40, 41, 42]
ctreader = ctfishpy.CTreader()

ct, stack_metadata = ctreader.read(fish_numbers[0], r = (0,100))
label = ctreader.read_label(labelpath)
#ctreader.view(ct)

#ct = np.array([x.T for x in ct])



d = datagen.apply_transform(ct.T, data_gen_args)
d = np.array(d)

ctreader.view(d, thresh = True)


