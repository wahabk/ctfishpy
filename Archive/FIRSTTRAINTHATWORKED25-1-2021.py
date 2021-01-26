from ctfishpy.model.dataGenie import *
import matplotlib.pyplot as plt
import numpy as np
import time
import segmentation_models as sm
from segmentation_models import Unet
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
timestr = time.strftime("%Y-%m-%d-%H-%M")



def lr_scheduler(epoch, learning_rate):
    decay_rate = 1
    decay_step = 13
    if epoch % decay_step == 0 and epoch != 0 and epoch != 1:
        return learning_rate * decay_rate
    return learning_rate

data_gen_args = dict(rotation_range=10, # degrees
                    width_shift_range=10, #pixels
                    height_shift_range=10,
                    shear_range=5, #degrees
                    zoom_range=0.1, # up to 1
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)


sample = [200,218,240,277,330,337,341,462,464, 40, 78, 364]
val_sample = [40, 78, 364]
val_steps = 8
batch_size = 64
steps_per_epoch = 34
epochs = 100
lr = 1e-5
BACKBONE = 'resnet34'
weights = 'imagenet'
weightspath = 'output/Model/unet_checkpoints.hdf5'
# 'output/Model/unet_checkpoints.hdf5'
opt = Adam(learning_rate=lr)
shape=224
input_shape = (shape,shape,1)
encoder_freeze=True
classes = 4
validation_split=0.3

model_checkpoint = ModelCheckpoint(weightspath, monitor = 'loss', verbose = 1, save_best_only = True)


callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
    model_checkpoint
]


dice_loss = sm.losses.DiceLoss(class_weights=np.array([1,5,1,5])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


xtrain, ytrain, sample_weights = dataGenie(batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

print(xtrain.shape, np.amax(xtrain), np.mean(xtrain))
print(ytrain.shape, np.amax(ytrain), np.mean(ytrain))
# xval, yval, sample_weights = dataGenie(batch_size = batch_size,
#                         data_gen_args = dict(),
#                         fish_nums = val_sample)


base_model = Unet(BACKBONE, encoder_weights=weights, classes=classes, activation='softmax', encoder_freeze=encoder_freeze)
inp = Input(shape=(shape, shape, 1))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)

#model.load_weights(weights)
model.compile(optimizer=opt, loss=total_loss, metrics=[])
history = model.fit(xtrain, y=ytrain, validation_split=validation_split, steps_per_epoch = steps_per_epoch, 
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

