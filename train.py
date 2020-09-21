from ctfishpy.unet.model import *
from ctfishpy.dataGenie import *
import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0,
                    width_shift_range=5,
                    height_shift_range=5,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='constant',
                    cval = 0)

sample = [76, 40, 81, 85, 88, 218, 222, 236, 298, 425]
batch_size = 10
steps_per_epoch = 1
epochs = 1

datagenie = DataGenie(  batch_size = batch_size,
                        data_gen_args = data_gen_args,
                        fish_nums = sample)

'''
training generator from z
trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1)
'''

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

model = unet()
model_checkpoint = ModelCheckpoint(f'output/Model/{timestr}_unet_checkpoints.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
history = model.fit_generator(datagenie, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks = [model_checkpoint])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose = 1) # read about this one
saveResult("output/Model/Test/", results)

