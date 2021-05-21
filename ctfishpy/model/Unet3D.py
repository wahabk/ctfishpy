from ..controller import CTreader, cc
import matplotlib.pyplot as plt
import numpy as np
import time
import json, codecs, pickle, csv
import segmentation_models_3D as sm3d
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
sm3d.set_framework('tf.keras')
from .Unet import *
from .generator import customImageDataGenerator

class Unet3D(Unet):
	def __init__(self, organ):
		super().__init__(organ)
		self.shape = (128,320,160,1)
		self.pretrain = False
		self.weightsname = 'unet3d_checkpoints'
		self.weights = 'imagenet'
		self.batch_size = 1
		self.alpha = 0.7

	def getModel(self):
		self.weightspath = 'output/Model/'+self.weightsname+'.hdf5'
		
		if self.pretrain:
			self.weights = 'imagenet'
		else:
			self.weights = None
		
		optimizer = Adam()
		optimizer.learning_rate = self.lr
		model = sm3d.Unet(self.BACKBONE, input_shape=self.shape, encoder_weights=self.weights, classes=self.nclasses, activation=self.activation, encoder_freeze=self.encoder_freeze)
		

		if self.rerun: model.load_weights(self.weightspath)
		model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
		# self.loss = self.multi_class_tversky_loss.__name__
		self.loss = self.loss.__name__
		return model

	def train(self, sample, val_sample, test_sample=None):
		self.weightspath = 'output/Model/'+self.weightsname+'.hdf5'
		self.sample = sample
		self.val_sample = val_sample
		self.test_sample = test_sample
		self.steps_per_epoch = int(len(self.sample)/ self.batch_size)
		self.val_steps = int(len(self.val_sample)/ self.batch_size)
		members = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
		print(members)
		
		self.trainStartTime = time.strftime("%Y-%m-%d-%H-%M") #save time that you started training
		data_gen_args = dict(zoom_range=0.1,
				horizontal_flip=True,
				vertical_flip = True,
				fill_mode='constant',
				cval = 0)

		
		datagenie = 	self.dataGenie(batch_size=self.batch_size,
						data_gen_args = data_gen_args,
						fish_nums = self.sample)

		valdatagenie = 	self.dataGenie(batch_size = self.batch_size,
						data_gen_args = dict(),
						fish_nums = self.val_sample)
		
		model = self.getModel()
		
		callbacks = [
			ModelCheckpoint(self.weightspath, monitor = 'loss', verbose = 1, save_best_only = True),
			#keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
			TerminateOnBaseline('val_f1-score', baseline=0.80)
		]
		
		history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = self.steps_per_epoch, 
						epochs = self.epochs, callbacks=callbacks, validation_steps=self.val_steps)
		

		if self.test_sample:
			testgenie = self.dataGenie(batch_size = self.batch_size,
						data_gen_args = dict(),
						fish_nums = self.test_sample)
			self.score = model.evaluate(testgenie, batch_size=self.batch_size, steps=int(len(self.test_sample)*self.roiZ / self.batch_size))
			print(self.score)
		else:
			h = history.history
			loss = h['loss']
			valf1 = h['val_f1-score']
			best_val = valf1[loss.index(min(loss))]
			self.score=[min(loss), best_val]
			print(f'\n\n Score best loss, val-f1 :{self.score}')

		self.saveParams()
		self.history = history.history
		self.history['time'] = self.trainStartTime
		self.saveHistory(f'output/Model/History/{self.trainStartTime}_history.json', self.history)

	def dataGenie(self, batch_size, data_gen_args, fish_nums, shuffle=True):
		imagegen = customImageDataGenerator(**data_gen_args)
		maskgen = customImageDataGenerator(**data_gen_args)
		
		ctreader = CTreader()

		centres_path = ctreader.centres_path
		with open(centres_path, 'r') as fp:
			centres = json.load(fp)

		roiZ=self.shape[0]
		roiSize=self.shape[1]
		seed = self.seed

		ct_list, label_list = [], []
		for num in fish_nums:
			center = centres[str(num)]
			z_center = center[0] # Find center of cc result and only read roi from slices
			ct, stack_metadata = ctreader.read(num, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)

			# train on manual and auto
			if num in manuals:
				organ = 'Otoliths'
				align = True if num in [78,200,218,240,277,330,337,341,462,464,364,385] else False
				is_amira = True
			elif num not in manuals:
				organ = 'Otoliths-unet'
				align = False
				is_amira = False
									
			
			label = ctreader.read_label('Otoliths', n=num,  align=align, is_amira=True)
			label = ctreader.crop_around_center3d(label, center = center, roiSize=roiSize, roiZ=roiZ)
			center[0] = int(roiZ/2) # Change center to 0 because only read necessary slices but cant do that with labels since hdf5
			ct = ctreader.crop_around_center3d(ct, center = center, roiSize=roiSize, roiZ=roiZ)

			if label.shape != ct.shape:
				raise Exception('X and Y shapes are different')

			new_mask = np.zeros(label.shape + (self.nclasses,))
			for i in range(self.nclasses):
				#for one pixel in the image, find the class in mask and convert it into one-hot vector
				new_mask[label == i, i] = 1
			mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2],new_mask.shape[3]))
			label = mask

			ct_list.append(ct)
			label_list.append(label)
			ct, label = None, None
			
		
		ct_list = np.array(ct_list, dtype='float32')
		label_list = np.array(label_list, dtype='float32')
		ct_list      = ct_list[:,:,:,:,np.newaxis] # add final axis to show datagens its grayscale

		print('[dataGenie] Initialising image and mask generators')

		image_generator = imagegen.flow(ct_list,
			batch_size = batch_size,
			#save_to_dir = 'output/Keras/',
			# save_prefix = 'dataGenie',
			seed = seed,
			shuffle=shuffle,
			)
		mask_generator = maskgen.flow(label_list, 
			batch_size = batch_size,
			seed = seed,
			shuffle=shuffle
			)
		
		datagen = zip(image_generator, mask_generator)
		for x_batch, y_batch in datagen:
			x_batch = x_batch/np.max(x_batch)
			y_batch = y_batch/np.max(y_batch)
			y_batch = np.around(y_batch, decimals=0)
			# print(x_batch[0].shape, x_batch[0].dtype, np.amax(x_batch[0]))
			# print(y_batch[0].shape, y_batch[0].dtype, np.amax(y_batch[0]))
			yield (x_batch, y_batch)


	def multi_class_tversky(self, y_true, y_pred):
		smooth = 1
		alpha=self.alpha

		y_true = K.permute_dimensions(y_true, (4,3,2,1,0))
		y_pred = K.permute_dimensions(y_pred, (4,3,2,1,0))

		y_true_pos = K.batch_flatten(y_true)
		y_pred_pos = K.batch_flatten(y_pred)
		true_pos = K.sum(y_true_pos * y_pred_pos, 1)
		false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
		false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
		alpha = 0.7
		return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

	def multi_class_tversky_loss(self, y_true, y_pred):
		return (1-self.multi_class_tversky(y_true, y_pred))

