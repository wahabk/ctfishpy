import ctfishpy
import torch
from ctfishpy.train_utils import CTDataset2D, undo_one_hot, renormalise
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
import random
import albumentations as A

if __name__ == "__main__":
	dataset_path = '/home/ak18001/Data/HDD/uCT'
	ctreader = ctfishpy.CTreader(dataset_path)

	bone = 'OTOLITHS'
	roiSize = (128,128,160)
	batch_size = 1
	n_classes = 4
	num_workers = 1
	all_keys = [1, 39, 64, 74, 96, 98, 112, 113, 115, 133, 186, 193, 197, 220, 241, 275, 276, 295, 311, 313, 314, 315, 316, 371, 374, 420, 427]
	random.shuffle(all_keys)
	check_set = all_keys[:1]
	print(f"checking dataset {check_set}")

	transforms = A.Compose([
		A.Flip(p=0.25),
		A.Affine(p=0.25),
		A.GaussianBlur(p=0.3),
		A.RandomBrightnessContrast(p=0.4),
		A.GaussNoise(var_limit=(0.001,0.01), p=0.25),
		A.RandomGamma(p=0.5),
	])

	check_ds = CTDataset2D(dataset_path, bone, check_set, roiSize, n_classes=n_classes, transform=transforms) 
	check_loader = torch.utils.data.DataLoader(check_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
	


	while True:
		x, y = next(iter(check_loader))
		
		# print(example[0].shape)
		# print(example[1].shape)

		print(x.shape, x.max(), x.min())
		print(y.shape, y.max(), y.min())
		print(type(x))

		x = x.cpu().numpy()  # send to cpu and transform to numpy.ndarray
		y = torch.squeeze(y).cpu().numpy()

		x = renormalise(x)
		y = undo_one_hot(y, n_classes=n_classes)
		# y = y[3]

		print(x.shape, x.max(), x.min())
		print(y.shape, y.max(), y.min())
		print(type(x))

		ctreader.view(x, label=y)

		# x = x/x.max()
		# histogram, bin_edges = np.histogram(x, bins=256, range=(0, 1))
		# plt.figure()
		# plt.title("Grayscale Histogram")
		# plt.xlabel("grayscale value")
		# plt.ylabel("pixel count")
		# plt.xlim([0.0, 1.0])  # <- named arguments do not work here
		# plt.plot(bin_edges[0:-1], histogram)  # <- or here

		plt.show()

		# exit()

		# array_projection = np.max(x, axis=0)
		# label_projection = np.max(y, axis=0)
		# sidebyside = np.concatenate((array_projection, label_projection), axis=1)
		# sidebyside /= sidebyside.max()
		# plt.imsave('output/test_genie.png', sidebyside, cmap='gray')