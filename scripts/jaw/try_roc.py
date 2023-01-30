import ctfishpy
import napari
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_auc_roc(fpr, tpr, auc_roc):
	fig, ax = plt.subplots(1,1)
	ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
	ax.plot([0, 1], [0, 1], 'k--')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('Receiver operating characteristic')
	ax.legend(loc="lower right")

	return fig


if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"

	ctreader = ctfishpy.CTreader(dataset_path)

	bone = "JAW"
	# name = "JAW_manual"
	# name  = "JAW_20221208"
	name  = "JAW_20230101"
	new_name  = "JAW_20230124"
	roiSize = (256, 256, 320)
	keys = ctreader.get_hdf5_keys(dataset_path+f"/LABELS/JAW/{new_name}.h5")
	print(len(keys), keys)


	index = 1
	true_label = ctreader.read_label(bone, index, name=name)

	center = ctreader.jaw_centers[index]
	true_label = ctreader.crop3d(true_label, roiSize, center)

	
	pred_label = true_label
	# pred_label[true_label==3]=0
	print(pred_label.shape)

	true_label = torch.from_numpy(true_label)
	y = F.one_hot(true_label.to(torch.int64), 5)
	y = y.permute([3,0,1,2]) # permute one_hot to channels first after batch
	print(y.shape)
	y = y.squeeze().to(torch.float32)
	true_label = y

	pred_label = torch.from_numpy(pred_label)
	y = F.one_hot(pred_label.to(torch.int64), 5)
	y = y.permute([3,0,1,2]) # permute one_hot to channels first after batch
	y = y.squeeze().to(torch.float32)
	pred_label = y

	aucroc = roc_auc_score(true_label.flatten(), pred_label.flatten(), average='weighted', multi_class='ovo')
	print(aucroc)

	true_vector = np.array(true_label, dtype='uint8').flatten()
	pred_vector = np.array(pred_label, dtype='uint8').flatten()
	fpr, tpr, _ = roc_curve(true_vector, pred_vector)
	fig = plot_auc_roc(fpr, tpr, aucroc)
	print(fpr, tpr)

	plt.show()