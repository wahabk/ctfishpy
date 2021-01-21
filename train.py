from ctfishpy.model.dataGenie import *
import matplotlib.pyplot as plt
import numpy as np
import ctfishpy
import time
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import FScore
import tensorflow.keras as keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Conv2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
timestr = time.strftime("%Y-%m-%d")

def lr_scheduler(epoch, learning_rate):
    decay_rate = 0.1
    decay_step = 10
    if epoch % decay_step == 0 and epoch not in [0, 1]:
        return learning_rate * decay_rate
    return learning_rate

data_gen_args = dict(rotation_range=0.01,
                    width_shift_range=0.01,
                    height_shift_range=0.09,
                    shear_range=0.09,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)


sample = [200,218,240,277,330,337,341,462,464,364]
val_sample = [40, 78]
val_steps = 8
batch_size = 64
steps_per_epoch = 23
epochs = 50
lr = 1e-5
BACKBONE = 'resnet34'
weights = 'imagenet'# 'output/Model/unet_checkpoints.hdf5'
opt = Adam(learning_rate=lr)

model_checkpoint = ModelCheckpoint(f'output/Model/unet_checkpoints.hdf5', 
                                        monitor = 'loss', verbose = 1, save_best_only = True)

callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
    model_checkpoint
]

datagenie = dataGenie(batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

valdatagenie = dataGenie(batch_size = batch_size,
                        data_gen_args = dict(),
                        fish_nums = val_sample)




base_model = Unet(BACKBONE, encoder_weights=weights, classes=4, activation='sigmoid')
inp = Input(shape=(128, 128, 1))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)

# model.load_weights(weights)
model.compile(optimizer=opt, loss=dice_loss, metrics=[dice_loss])
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

