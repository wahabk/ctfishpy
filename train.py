from ctfishpy.model.dataGenie import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
import ctfishpy
import os
import time
from segmentation_models import Unet
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Conv2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
timestr = time.strftime("%Y-%m-%d")

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection +smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 8
    if epoch % decay_step == 0 and epoch not in [0, 1]:
        return lr * decay_rate
    return lr

data_gen_args = dict(rotation_range=0.01,
                    width_shift_range=0.01,
                    height_shift_range=0.09,
                    shear_range=0.09,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

sample = [81, 85, 88, 222, 425, 236, 218]
val_sample = [40, 76]
val_steps = 8
batch_size = 8
steps_per_epoch = 64
epochs = 10
lr = 1e-4

datagenie = dataGenie(batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

valdatagenie = dataGenie(batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = val_sample)


model_checkpoint = ModelCheckpoint(f'output/Model/unet_checkpoints.hdf5', 
                                        monitor = 'loss', verbose = 1, save_best_only = True)

callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
    model_checkpoint
]

opt = Adam(lr=lr)
base_model = Unet('resnet34', encoder_weights=None, input_shape=(128, 128, 3), classes=1, activation='sigmoid')
inp = Input(shape=(None, None, 1))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)

model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = steps_per_epoch, 
                    epochs = epochs, callbacks=callbacks, validation_steps=val_steps)


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Unet-otolith loss (lr={lr})')
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

