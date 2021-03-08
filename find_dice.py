import ctfishpy
import numpy as np
import matplotlib.pyplot as plt 

def dice(y, yhat, k=1):
	return np.sum(yhat[y==k]==k)*2.0 / (np.sum(yhat[yhat==k]==k) + np.sum(y[y==k]==k))

# plt.imshow(gt)
# plt.show()
# plt.imshow(seg)
# plt.show()

if __name__ == '__main__':

	ctreader = ctfishpy.CTreader()
	n = 256
	gt = ctreader.read_label('Otoliths_Mariele', n, align=False, is_amira=True)
	gt[gt==2]=1
	gt[gt==3]=2
	gt[gt==4]=3
	pr = ctreader.read_label('Otoliths_Zac', n, align=False, is_amira=True)
	print(gt.shape, pr.shape)

	# ctreader.view(gt, label=gt)
	# ctreader.view(pr, label = pr)

	for i in range(0,4):
		d = dice(gt, pr, k=i)
		print(f'Dice similarity score for class {i} is {d}')


[40, 242, 256,421,423]