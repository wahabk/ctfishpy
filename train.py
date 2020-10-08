from ctfishpy.unet.model import *
from ctfishpy.dataGenie import *
import ctfishpy
import os
import time
timestr = time.strftime("%Y-%m-%d")
import datetime
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras



data_gen_args = dict(rotation_range=0.01,
                    width_shift_range=0.01,
                    height_shift_range=0.09,
                    shear_range=0.09,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

sample = [76, 81, 85, 88, 222, 425, 236, 218]
val_sample = [40]
val_steps = 8
batch_size = 8
steps_per_epoch = 64
epochs = 1024

datagenie = dataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

valdatagenie = dataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = val_sample)


model_checkpoint = ModelCheckpoint(f'output/Model/unet_checkpoints.hdf5', 
                                        monitor = 'loss', verbose = 1, save_best_only = True)

callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
    model_checkpoint
]

unet = Unet()
model = unet.get_unet(preload=True)  #unet() 
history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = steps_per_epoch, 
                    epochs = epochs, callbacks=callbacks, validation_steps=val_steps)


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Unet-otolith loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'output/Model/loss_curves/{timestr}_loss.png')

# import pdb; pdb.set_trace()
# testGenie = testGenie(test_num)
# results = model.predict(testGenie, 16, verbose = 1) # read about this one
# output = fixFormat(results, label=True)
# lumpfish = Lumpfish()
# lumpfish.write_label('prediction.hdf5', output)

