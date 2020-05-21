from ctfishpy.unet.model import *
from ctfishpy.unet.dataGenie import *
import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=180,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.2,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

sample = [40, 76, 81, 85, 88, 218, 222, 236, 298, 425]
batch_size = 8

datagenie = DataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

'''
trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1)
'''

model = unet()
model_checkpoint = ModelCheckpoint(f'output/Model/{timestr}unet_checkpoints.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
model.fit_generator(datagenie, steps_per_epoch = 100, epochs = 2, callbacks = [model_checkpoint])

#testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose = 1) # read about this one
saveResult("output/Model/Test/", results)

