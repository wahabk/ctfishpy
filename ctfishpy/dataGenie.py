from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .CTreader import CTreader
from .cc import cc
import gc
import cv2


def fixFormat(batch, label = False):
    # change format of image batches to make viewable with ctreader
    if not label: return np.squeeze(batch.astype('uint16'), axis = 3)
    if label: return np.squeeze(batch.astype('uint8'), axis = 3)

def DataGenie(batch_size, data_gen_args, fish_nums = None):
    imagegen = ImageDataGenerator(**data_gen_args)
    maskgen = ImageDataGenerator(**data_gen_args)

    while True:
        for num in fish_nums:
            scan_halves = None
            templatePath = './Data/Labels/CC/otolith_template_10.hdf5'

            ctreader = CTreader()
            ct, stack_metadata = ctreader.read(num, r = None, align=True)#(1400,1600))
            label = ctreader.read_label(f'../../Data/HDD/uCT/Labels/Otolith1/{num}.h5', n=num,  align=True, manual=True)
            template = ctreader.read_label(templatePath, manual=False)
            projections = ctreader.get_max_projections(num)

            center, error = cc(num, template, thresh=200, roiSize=50)
            ct = ctreader.crop_around_center3d(ct, center = center, roiSize=257, roiZ=100)
            label = ctreader.crop_around_center3d(label, center = center, roiSize=257, roiZ=100)


            ct      = ct[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
            label   = label[:,:,:,np.newaxis] # add final axis to show datagens its grayscale

            print('[dataGenie] Initialising image and mask generators')

            image_generator = imagegen.flow(ct,
                batch_size = batch_size,
                #save_to_dir = 'output/Keras/',
                save_prefix = 'dataGenie',
                seed = 420
                )
            mask_generator = maskgen.flow(label, 
                batch_size = batch_size,
                #save_to_dir = 'output/Keras/',
                save_prefix = 'dataGenie',
                seed = 420
                )
            print('Ready.')

            datagen = zip(image_generator, mask_generator)
            for x_batch, y_batch in datagen:
                yield (x_batch, y_batch)
            
            ct, label = None, None
            gc.collect()

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
