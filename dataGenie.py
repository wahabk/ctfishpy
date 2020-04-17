from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import ctfishpy
import gc
import cv2

def fixFormat(batch, label = False):
    # change format of image batches to make viewable with ctreader
    if not label: return np.squeeze(batch.astype('uint16'), axis = 3)
    if label: return np.squeeze(batch.astype('uint8'), axis = 3)


def DataGenie(batch_size, data_gen_args, labelpath, fish_nums = None):
    imagegen = ImageDataGenerator(**data_gen_args)
    maskgen = ImageDataGenerator(**data_gen_args)

    ctreader = ctfishpy.CTreader()
    ct, stack_metadata = ctreader.read(40, r = None)#(1400,1600))
    label = ctreader.read_label(labelpath)

    ct = np.array([cv2.resize(slice_, (256,256)) for slice_ in ct])
    label = np.array([cv2.resize(slice_, (256,256)) for slice_ in label])
    ct = ct[:,:,:,np.newaxis] # add final axis to show datagen its grayscale
    label = label[:,:,:,np.newaxis]

    image_generator = imagegen.flow(ct[1360:1380],
        batch_size = batch_size,
        #save_to_dir = 'output/Keras/',
        save_prefix = 'dataGenie',
        seed = 420
        )
    mask_generator = maskgen.flow(label[1360:1380], 
        batch_size = batch_size,
        #save_to_dir = 'output/Keras/',
        save_prefix = 'dataGenie',
        seed = 420
        )

    ct = None
    gc.collect()

    datagen = zip(image_generator, mask_generator)
    for x_batch, y_batch in datagen:
        yield (x_batch, y_batch)


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
labelpath = '../../Data/HDD/uCT/Labels/Otolith1/040.h5'

datagenie = DataGenie(batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        labelpath = labelpath)

ctreader = ctfishpy.CTreader()

for x_batch, y_batch in datagenie:
    #print(x_batch.shape)
    #print(y_batch.shape)
    x_batch = fixFormat(x_batch)  # remove last weird axis
    y_batch = fixFormat(y_batch, label = True)  # remove last weird axis

    ctreader.view(x_batch, label = y_batch)

    #break