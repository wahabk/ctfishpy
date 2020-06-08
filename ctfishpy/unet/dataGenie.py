from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .. import CTreader
import gc
import cv2
sample = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]
# change label path to read labels directly

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

            ctreader = CTreader()
            ct, stack_metadata = ctreader.read(num, r = None)#(1400,1600))
            label = ctreader.read_label(f'../../Data/HDD/uCT/Labels/Otolith1/{num}.h5')

            ct      = np.array([cv2.resize(slice_, (256,256)) for slice_ in ct])
            label   = np.array([cv2.resize(slice_, (256,256)) for slice_ in label])
            ct      = ct[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
            label   = label[:,:,:,np.newaxis] # add final axis to show datagens its grayscale

            print('[dataGenie] Initialising image and mask generators')

            scan_length = ct.shape[0]

            scan_halves = [ [ct[0:1000], label[0:1000]], 
                            [ct[1000:scan_length], label[1000:scan_length]]]

            ct, label = None, None
            gc.collect()

            for x, y in scan_halves:
                image_generator = imagegen.flow(x,
                    batch_size = batch_size,
                    #save_to_dir = 'output/Keras/',
                    save_prefix = 'dataGenie',
                    seed = 420
                    )
                mask_generator = maskgen.flow(y, 
                    batch_size = batch_size,
                    #save_to_dir = 'output/Keras/',
                    save_prefix = 'dataGenie',
                    seed = 420
                    )
                print('Ready.')

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

datagenie = DataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

# ctreader = ctfishpy.CTreader()

# for x_batch, y_batch in datagenie:
#     #print(x_batch.shape)
#     #print(y_batch.shape)
#     x_batch = fixFormat(x_batch)  # remove last weird axis
#     y_batch = fixFormat(y_batch, label = True)  # remove last weird axis

#     ctreader.view(x_batch, label = y_batch)

#     #break
