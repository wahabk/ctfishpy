# https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
from ..controller import CTreader, cc
import matplotlib.pyplot as plt
import numpy as np
import time
import json, codecs, pickle, csv
import segmentation_models as sm
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sm.set_framework('tf.keras')

class Unet():
	def __init__(self, organ):
		self.shape = (224,224)
		self.roiZ = 150
		self.organ = organ
		self.batch_size = 32
		self.epochs = 30
		self.lr = 1e-4
		self.pretrain = True #write this into logic
		self.BACKBONE = 'resnet34'
		self.weightspath = 'output/Model/unet_checkpoints.hdf5'
		self.encoder_freeze=True
		self.nclasses = 3
		self.activation = 'softmax'
		self.class_weights = np.array([0.5,1.25,1.5])
		self.metrics = [sm.metrics.FScore(), sm.metrics.IOUScore()]
		self.rerun = False
		self.slice_weighting = 1
		self.alpha = 0.7
		self.loss=sm.losses.CategoricalCELoss()
		self.seed=69
		# 'output/Model/unet_checkpoints.hdf5'
		members = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
		print(members)
		

	def saveParams(self):
		members = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
		with open('output/Model/trainingParameters.csv', 'a', newline='') as f:  
			w = csv.DictWriter(f, members.keys())
			w.writerow(members)

	def getModel(self):

		dice_loss = sm.losses.DiceLoss(class_weights=self.class_weights) 
		if self.pretrain:
			self.weights = 'imagenet'
		else:
			self.weights = None
		optimizer = Adam(learning_rate=self.lr)

		base_model = sm.Unet(self.BACKBONE, encoder_weights=self.weights, classes=self.nclasses, activation=self.activation, encoder_freeze=self.encoder_freeze)
		#base_model = sm.Unet(self.BACKBONE, encoder_weights=None, classes=self.nclasses, activation=self.activation, encoder_freeze=False)
		inp = Input(shape=(self.shape[0], self.shape[1], 1))
		l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
		out = base_model(l1)
		model = Model(inp, out, name=base_model.name)
		
		if self.rerun: model.load_weights(self.weightspath)
		model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
		# self.loss = self.multi_class_tversky_loss.__name__
		self.loss = self.loss.__name__
		return model

	def train(self, sample, val_sample, test_sample=None):
		self.sample = sample
		self.val_sample = val_sample
		self.test_sample = test_sample
		
		
		self.trainStartTime = time.strftime("%Y-%m-%d-%H-%M") #save time that you started training
		data_gen_args = dict(rotation_range=10, # degrees
					width_shift_range=10, #pixels
					height_shift_range=10,
					shear_range=10, #degrees
					zoom_range=0.1, # up to 1
					horizontal_flip=True,
					vertical_flip = True,
					fill_mode='constant',
					cval = 0)

		
		datagenie = self.dataGenie(batch_size=self.batch_size,
						data_gen_args = data_gen_args,
						fish_nums = self.sample)

		valdatagenie= self.dataGenie(batch_size = self.batch_size,
						data_gen_args = dict(),
						fish_nums = self.val_sample)
		
		model = self.getModel()
		self.steps_per_epoch = int(len(self.sample)*self.roiZ / self.batch_size)
		self.val_steps = int(len(self.val_sample)*self.roiZ / self.batch_size)

		model_checkpoint = ModelCheckpoint(self.weightspath, monitor = 'loss', verbose = 1, save_best_only = True)


		callbacks = [
			keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0),
			model_checkpoint
		]
		
		history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = self.steps_per_epoch, 
						epochs = self.epochs, callbacks=callbacks, validation_steps=self.val_steps)
		

		if self.test_sample:
			testgenie = self.dataGenie(batch_size = self.batch_size,
						data_gen_args = dict(),
						fish_nums = self.test_sample)
			self.score = model.evaluate(testgenie, batch_size=self.batch_size, steps=int(len(self.test_sample)*self.roiZ / self.batch_size))
			print(self.score)
		
		self.saveParams()
		
		self.history = history.history
		self.history['time'] = self.trainStartTime
		self.saveHistory(f'output/Model/History/{self.trainStartTime}_history.json', self.history)



	def makeLossCurve(self, history=None):
		if history == None: history = self.history

		metrics = ['loss','val_loss','f1-score','val_f1-score']
		for m in metrics:
			plt.plot(history[m])
		
		plt.title(f'Unet-otolith loss (lr={self.lr})')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.ylim(0,1)
		plt.legend(metrics, loc='upper left')
		plt.savefig('output/Model/loss_curves/'+history['time']+'_loss.png')

	def predict(self, n):
		base_model = sm.Unet(self.BACKBONE, classes=self.nclasses, activation=self.activation, encoder_freeze=self.encoder_freeze)
		inp = Input(shape=(self.shape[0], self.shape[1], 1))
		l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
		out = base_model(l1)
		model = Model(inp, out, name=base_model.name)
		model.load_weights(self.weightspath)
		
		# base_model = sm.Unet(self.BACKBONE, encoder_weights=None, classes=self.nclasses, activation=self.activation, encoder_freeze=True)
		# inp = Input(shape=(self.shape[0], self.shape[1], 1))
		# l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
		# out = base_model(l1)


		test = self.testGenie(n)
		results = model.predict(test, self.batch_size) # read about this one

		label = np.zeros(results.shape[:-1], dtype = 'uint8')
		for i in range(self.nclasses):
			result = results[:, :, :, i]
			label[result>0.5] = i
		
		ct = np.squeeze((test).astype('float32'), axis = 3)
		ct = np.array([_slice * 255 for _slice in ct], dtype='uint8') # Normalise 16 bit slices

		return label, ct
	
	def dataGenie(self, batch_size, data_gen_args, fish_nums, shuffle=True):
		imagegen = ImageDataGenerator(**data_gen_args, rescale = 1./65535)
		maskgen = ImageDataGenerator(**data_gen_args)
		ctreader = CTreader()
		template = ctreader.read_label(self.organ, 0)


		centres_path = ctreader.centres_path
		with open(centres_path, 'r') as fp:
			centres = json.load(fp)

		roiZ=self.roiZ
		roiSize=self.shape[0]
		seed = self.seed

		ct_list, label_list = [], []
		for num in fish_nums:
			# take out cc for now
			# center = cc(num, template, thresh=80, roiSize=224)
			center = centres[str(num)]
			z_center = center[0] # Find center of cc result and only read roi from slices

			ct, stack_metadata = ctreader.read(num, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)
									
			align = True if num in [40,78,200,218,240,277,330,337,341,462,464,364,385] else False
			label = ctreader.read_label('Otoliths', n=num,  align=align, is_amira=True)
			
			label = ctreader.crop_around_center3d(label, center = center, roiSize=roiSize, roiZ=roiZ)
			center[0] = int(roiZ/2) # Change center to 0 because only read necessary slices but cant do that with labels since hdf5
			ct = ctreader.crop_around_center3d(ct, center = center, roiSize=roiSize, roiZ=roiZ)

			if label.shape != ct.shape:
				raise Exception('X and Y shapes are different')

				

			if self.organ == 'Otoliths':
				# remove utricular otoliths
				label[label == 2] = 0
				label[label == 3] = 2

			new_mask = np.zeros(label.shape + (self.nclasses,))
			for i in range(self.nclasses):
				#for one pixel in the image, find the class in mask and convert it into one-hot vector
				new_mask[label == i,i] = 1
			
			mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2],new_mask.shape[3]))
			label = mask
			ct_list.append(ct)
			label_list.append(label)
			ct, label = None, None
			
		
		ct_list = np.vstack(ct_list)
		label_list = np.vstack(label_list)
		ct_list = np.array(ct_list, dtype='float32')
		label_list = np.array(label_list, dtype='float32')
		ct_list      = ct_list[:,:,:,np.newaxis] # add final axis to show datagens its grayscale

		print('[dataGenie] Initialising image and mask generators')

		image_generator = imagegen.flow(ct_list,
			batch_size = batch_size,
			#save_to_dir = 'output/Keras/',
			save_prefix = 'dataGenie',
			seed = seed,
			shuffle=shuffle,
			)
		mask_generator = maskgen.flow(label_list, 
			batch_size = batch_size,
			#save_to_dir = 'output/Keras/',
			save_prefix = 'dataGenie',
			seed = seed,
			shuffle=shuffle
			)
		
		print('[dataGenie] Ready... Extracting data')

		train_generator = zip(image_generator, mask_generator)
		for (img,mask) in train_generator:
			yield (img,mask)

		# #extract data frin generatirs
		# test_batches = [image_generator, mask_generator]
		# xdata, ydata = [], []
		# for i in range(0,int(len(ct_list)*self.epochs/batch_size)):
		# 	xdata.extend(np.array(test_batches[0][i]))
		# 	ydata.extend(np.array(test_batches[1][i]))





	def testGenie(self, n):
		ctreader = CTreader()
		template = ctreader.read_label(self.organ, 0)
		# center = cc(n, template, thresh=80, roiSize=224)
		centres_path = ctreader.centres_path
		with open(centres_path, 'r') as fp:
			centres = json.load(fp)	
		center = centres[str(n)]
		z_center = center[0] # Find center of cc result and only read roi from slices
		roiZ=self.roiZ
		roiSize=self.shape[0]
		ct, stack_metadata = ctreader.read(n, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)#(1400,1600))
		center[0] = int(roiZ/2)
		ct = ctreader.crop_around_center3d(ct, center = center, roiSize=roiSize, roiZ=roiZ)
		ct = np.array([_slice / 65535 for _slice in ct], dtype='float32') # Normalise 16 bit slices
		ct = ct[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
		print(ct.shape)
		return ct

	def saveHistory(self, path, history):
		'''
		from https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
		'''
		with open(path, 'wb') as file_pi:
			pickle.dump(history, file_pi)

	def loadHistory(self, path):
		'''
		load history saved as pickle file

		parameters:
		path

		from https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
		'''
		with open(path, 'rb') as file_pi:
			history = pickle.load(file_pi)
		return history


	def multi_class_tversky(self, y_true, y_pred):
		smooth = 1
		alpha=self.alpha

		y_true = K.permute_dimensions(y_true, (3,1,2,0))
		y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

		y_true_pos = K.batch_flatten(y_true)
		y_pred_pos = K.batch_flatten(y_pred)
		true_pos = K.sum(y_true_pos * y_pred_pos, 1)
		false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
		false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
		return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

	def multi_class_tversky_loss(self, y_true, y_pred):
		return (1-self.multi_class_tversky(y_true, y_pred))

	def focal_multi_class_tversky_loss(self, y_true,y_pred):
		pt_1 = self.class_tversky(y_true, y_pred)
		gamma = 0.75
		return K.sum(K.pow((1-pt_1), gamma))

def lr_scheduler(epoch, learning_rate):
	decay_rate =  1
	decay_step = 12
	if epoch == decay_step and epoch != 0 and epoch != 1:
		return learning_rate * decay_rate
	return learning_rate

def fixFormat(batch, label = False):
	# change format of image batches to make viewable with ctreader
	if not label: return np.squeeze(batch.astype('uint16'), axis = 3)
	if label: return np.squeeze(batch.astype('uint8'), axis = 3)



		# # sample_weights and data extracted in case i need to focus 
		# # on slices that arent empty
		# print('[dataGenie] Finding data distrib')
		# xdata = np.array(xdata)
		# ydata = np.array(ydata)
		# total = ydata.shape[0]

		# temp = np.zeros(ydata.shape[:-1], dtype = 'uint8')
		# for i in [1,2]:
		# 	y = ydata[:, :, :, i]
		# 	temp[y==1] = i

		# distrib = [np.any(temp[i]) for i in range(total)]
		# distrib = np.array(distrib)
		# sample_weights = np.ones(shape=(total))
		# sample_weights[distrib == True] = self.slice_weighting
		
		# print(F'\n\n[dataGenie] Found full slices. distrib: {sum(distrib)}/{total} ratio: {sum(distrib)/total} sample_weights={sample_weights.shape[0]}')
		
		# xdata = xdata[distrib==True]#take only slices that have otoliths in them
		# ydata = ydata[distrib==True]
		# total = ydata.shape[0]
		# temp = np.zeros(ydata.shape[:-1], dtype = 'uint8')
		# for i in [1,2]:
		# 	y = ydata[:, :, :, i]
		# 	temp[y==1] = i
		# distrib = [np.any(temp[i]) for i in range(total)]
		# distrib = np.array(distrib)
		

		# print(F'\n\n[dataGenie] Done. Final distrib: {sum(distrib)}/{total} ratio: {sum(distrib)/total} sample_weights={sample_weights.shape[0]}')

		# return xdata, ydata, sample_weights