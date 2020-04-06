from unet import *
from dataGenie import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('testing git')
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGenerator = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

'''
trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1)
'''

model = unet()
model_checkpoint = ModelCheckpoint('unet_checkpoints.hdf5', monitor = 'loss', verbose = 1, save_best_only = True)
model.fit_generator(myGenerator, steps_per_epoch = 300, epochs = 1, callbacks = [model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose = 1) # read about this one
saveResult("data/membrane/test", results)
