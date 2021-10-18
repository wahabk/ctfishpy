from ..controller import CTreader
import numpy as np
import time
import segmentation_models_3D as sm3d
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from .Unet import *
from .generator import customImageDataGenerator
sm3d.set_framework('tf.keras')

class Unet3D(Unet):
	def __init__(self, organ):
		super().__init__(organ)
		self.shape = (128,288,128,1)
		self.pretrain = False
		self.weightsname = 'unet3d_checkpoints'
		self.weights = None
		self.encoder_freeze = False
		self.batch_size = 1
		self.alpha = 0.3
		self.BACKBONE = 'resnet18'
		self.metrics = [sm3d.metrics.FScore(threshold=0.3), sm3d.metrics.IOUScore()]
		self.loss = self.multiclass_tversky3d_loss

	def getModel(self):
		self.weightspath = 'output/Model/'+self.weightsname+'.hdf5'
		
		if self.pretrain:
			self.weights = 'imagenet'
		else:
			self.weights = None
		
		optimizer = Adam()
		optimizer.learning_rate = self.lr
		model = sm3d.Unet(self.BACKBONE, input_shape=self.shape, encoder_weights=None, classes=self.nclasses, activation=self.activation, encoder_freeze=self.encoder_freeze)
		
		if self.rerun: model.load_weights(self.weightspath)
		model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
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
			TerminateOnBaseline('val_f1-score', baseline=1)
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

	def predict(self, n, test_batch_size=1, thresh=0.5):
		self.weightspath = 'output/Model/'+self.weightsname+'.hdf5'

		model = sm3d.Unet(self.BACKBONE, input_shape=self.shape, encoder_weights=None, classes=self.nclasses, activation=self.activation, encoder_freeze=self.encoder_freeze)

		model.load_weights(self.weightspath)


		test, og_center, og_shape, og_ct = self.testGenie(n)
		print(np.shape(test))
		results = model.predict(test, test_batch_size) # read about this one

		label = np.zeros(results.shape[:-1], dtype = 'uint8')
		for i in range(self.nclasses):
			result = results[:, :, :, :, i]
			label[result>thresh] = i
		label = label[0]
		
		# ct = np.squeeze((test).astype('float32'), axis = 3)
		# ct = np.array([_slice * 255 for _slice in ct], dtype='uint8') # Normalise 16 bit slices

		# create empty stack with the size of original scan and insert label into original position
		new_stack = np.zeros(og_shape, dtype='uint8')
		z, x, y = og_center
		roiSize = label.shape
		print(label.shape, roiSize, og_center)
		
		zl = int(roiSize[0] / 2)
		xl = int(roiSize[1] / 2)
		yl = int(roiSize[2] / 2)
		
		new_stack[z - zl : z + zl, x - xl : x + xl, y - yl : y + yl] = label
		label = np.array(new_stack, dtype='uint8')
		return label, og_ct

	def dataGenie(self, batch_size, data_gen_args, fish_nums, shuffle=True):
		imagegen = customImageDataGenerator(**data_gen_args)
		maskgen = customImageDataGenerator(**data_gen_args)
		
		ctreader = CTreader()

		centres_path = ctreader.centres_path
		with open(centres_path, 'r') as fp:
			centres = json.load(fp)

		roiZ=self.shape[0]
		roiSize=self.shape[1:]
		seed = self.seed

		ct_list, label_list = [], []
		for num in fish_nums:
			center = centres[str(num)]
			z_center = center[0] # Find center of cc result and only read roi from slices
			ct, stack_metadata = ctreader.read(num, r = (z_center - int(roiZ/2), z_center + int(roiZ/2)), align=True)

			auto = [41,43,44,45,46,56,57,69,70,72,74,77,78,79,80,90,92,200,201,203]
			# train on manual and auto
			if num in auto:
				organ = 'Otoliths_unet2d'
				align = False
				is_amira = False
				
			elif num not in auto:
				organ = 'Otoliths'
				#fix for undergrads
				align = True if num in [78,200,218,240,277,330,337,341,462,464,364,385] else False
				is_amira = True
				
									
			
			label = ctreader.read_label(organ, n=num, is_amira=is_amira)
			label = ctreader.crop_around_center3d(label, center = center, roiSize=roiSize, roiZ=roiZ)
			center[0] = int(roiZ/2) # Change center to 0 because only read necessary slices but cant do that with labels since hdf5
			ct = ctreader.crop_around_center3d(ct, center = center, roiSize=roiSize, roiZ=roiZ)

			if label.shape != ct.shape:
				raise Exception('X and Y shapes are different')
			ct = ct[:,:,:,np.newaxis] # add final axis to show datagens its grayscale

			new_mask = np.zeros(label.shape + (self.nclasses,))
			for i in range(self.nclasses):
				#for one pixel in the image, find the class in mask and convert it into one-hot vector
				new_mask[label == i, i] = 1
			mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2],new_mask.shape[3]))
			label = mask

			if ct.shape[1] == 0: continue

			ct_list.append(ct)
			label_list.append(label)
			ct, label = None, None
			
		[print(c.shape) for c in ct_list]
		ct_list = np.array(ct_list, dtype='float32')
		print(ct_list.shape)
		label_list = np.array(label_list, dtype='float32')
		

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

	def testGenie(self, n):
		ctreader = CTreader()

		centres_path = ctreader.centres_path
		with open(centres_path, 'r') as fp:
			centres = json.load(fp)	
		center = centres[str(n)]

		ct, stack_metadata = ctreader.read(n, align=True)#(1400,1600))
		og_ct = ct.copy()
		og_shape = ct.shape
		og_center = center.copy()

		roiSize=self.shape[:-1]

		ct = ctreader.crop3d(ct, roiSize, center = center)
		ct = np.array([_slice / 65535 for _slice in ct], dtype='float32') # Normalise 16 bit slices
		ct = ct[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
		ct_list = np.array([ct])
		ct_list= ct_list.reshape((-1,)+self.shape)
		print(ct_list.shape)
		
		return ct_list, og_center, og_shape, og_ct

	def multiclass_tversky3d(self, y_true, y_pred):
		# https://github.com/nabsabraham/focal-tversky-unet/issues/3
		smooth = 1

		y_true = K.permute_dimensions(y_true, (4,1,2,3,0))
		y_pred = K.permute_dimensions(y_pred, (4,1,2,3,0))

		y_true_pos = K.batch_flatten(y_true)
		y_pred_pos = K.batch_flatten(y_pred)
		true_pos = K.sum(y_true_pos * y_pred_pos, 1)
		false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
		false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
		alpha = self.alpha
		return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

	def multiclass_tversky3d_loss(self, y_true, y_pred):
		return 1 - self.multiclass_tversky3d(y_true, y_pred)


