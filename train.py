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



data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    # vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

sample = [76, 40, 81, 85, 88, 222, 425, 236]
val_sample = [218]
val_steps = 8
batch_size = 16
steps_per_epoch = 25
epochs = 1024

datagenie = dataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

valdatagenie = dataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = val_sample)


unet = Unet()
model = unet.get_unet()  #unet()
model_checkpoint = ModelCheckpoint(f'output/Model/unet_checkpoints.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_steps=val_steps, callbacks = [model_checkpoint])

# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(f'output/Model/loss_curves/{timestr}_acc.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'output/Model/loss_curves/{timestr}_loss.png')

# import pdb; pdb.set_trace()
# testGenie = testGenie(test_num)
# results = model.predict(testGenie, 16, verbose = 1) # read about this one
# output = fixFormat(results, label=True)
# lumpfish = Lumpfish()
# lumpfish.write_label('prediction.hdf5', output)

