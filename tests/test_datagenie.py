import ctfishpy
import torch
from ctfishpy.train_utils import CTDataset, undo_one_hot, renormalise
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":

	ctreader = ctfishpy.CTreader()

	bone = 'Otoliths'
	roiSize = (128,128,256)
	batch_size = 1
	n_classes = 4
	num_workers = 10
	all_keys = [1, 39, 64, 74, 96, 98, 112, 113, 115, 133, 186, 193, 197, 220, 241, 275, 276, 295, 311, 313, 314, 315, 316, 371, 374, 420, 427]
	check_set = all_keys[:1]

	print(f"checking dataset {check_set}")

	transforms_affine = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.8),
		tio.RandomAffine(),
	])
	transforms_img = tio.Compose([
		tio.RandomAnisotropy(p=0.2),              # make images look anisotropic 25% of times
		tio.RandomBlur(p=0.2),
		tio.OneOf({
			tio.RandomNoise(0.1, 0.01): 0.1,
			tio.RandomBiasField(0.1): 0.1,
			tio.RandomGamma((-0.3,0.3)): 0.1,
			tio.RandomMotion(): 0.3,
		}),
		tio.RescaleIntensity((0.05,0.95)),
	])

	check_ds = CTDataset(bone, check_set, roiSize, n_classes=n_classes, transform=transforms_img, label_transform=transforms_affine, label_size=roiSize) 
	check_loader = torch.utils.data.DataLoader(check_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
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

	# array_projection = np.max(x, axis=0)
	# label_projection = np.max(y, axis=0)
	# sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	# sidebyside /= sidebyside.max()
	# plt.imsave('output/test_genie.png', sidebyside, cmap='gray')