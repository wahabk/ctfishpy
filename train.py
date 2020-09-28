from ctfishpy.unet.model import *
from ctfishpy.dataGenie import *
from ctfishpy import Lumpfish
import os
import time
timestr = time.strftime("%Y-%m-%d")
import datetime
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

sample = [76, 40, 81, 85, 88, 222, 236, 218, 425]
test_num = 218
val_sample = [218]
val_steps = 16
batch_size = 16
steps_per_epoch = 16
epochs = 64

datagenie = dataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

valdatagenie = dataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = val_sample)

'''
training generator from z
trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1)
'''

# logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                  histogram_freq = 1,
#                                                  profile_batch = '500,520')

unet2 = Unet2()
model = unet2.get_unet()  #unet()
model_checkpoint = ModelCheckpoint(f'output/Model/unet_checkpoints.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_steps=val_steps, callbacks = [model_checkpoint])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/Model/loss_curves/acc.png')
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('output/Model/loss_curves/loss.png')

testGenie = testGenie(test_num)
results = model.predict(testGenie, 100, verbose = 1) # read about this one
lumpfish = Lumpfish()
lumpfish.write_label('prediction.hdf5', results)

