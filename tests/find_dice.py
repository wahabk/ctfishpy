import ctfishpy
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



def dice(y, yhat, k=1):
	return np.sum(yhat[y==k]==k)*2.0 / (np.sum(yhat[yhat==k]==k) + np.sum(y[y==k]==k))

# plt.imshow(gt)
# plt.show()
# plt.imshow(seg)
# plt.show()

'''Making analysis'''
# if __name__ == '__main__':

# 	ctreader = ctfishpy.CTreader()
# 	sample = [40, 242, 256,421,423]
# 	scores = {}
# 	for n in sample:
# 		gt = ctreader.read_label('Otoliths_Mariele', n, align=False, is_amira=True)
# 		if n == 40:
# 			gt[gt==4]=1
# 			gt[gt==3]=5
# 			gt[gt==2]=3
# 			gt[gt==5]=2
# 		else:
# 			gt[gt==2]=1
# 			gt[gt==3]=2
# 			gt[gt==4]=3
# 		pr = ctreader.read_label('Otoliths_Zac', n, align=False, is_amira=True)
# 		print(gt.shape, pr.shape)

# 		# ctreader.view(gt, label=gt)
# 		# ctreader.view(pr, label = pr)
# 		dices = []
# 		for i in range(0,4):
# 			d = dice(gt, pr, k=i)
# 			print(f'Dice similarity score for class {i} is {d}')
# 			dices.append(d)
# 		scores[str(n)] = dices
# 	path = 'output/Zac&Mariel/manual_dice_scores.csv'
# 	df = pd.DataFrame.from_dict(scores, orient="index")
# 	df.to_csv(path)

''' This is for analysing and making figures '''
if __name__ == '__main__':
	path = 'output/Zac&Mariel/manual_dice_scores.csv'
	df = pd.read_csv(path)

	lag_acc = df.iloc[:]['1']
	utr_acc = df.iloc[:]['2']
	sag_acc = df.iloc[:]['3']

	# x = [lag_acc, utr_acc, sag_acc]
	df = df.drop('n', axis=1)
	df = df.drop('0', axis=1)
	df.rename(columns={"1": "Lagenal"}, inplace=True)
	df.rename(columns={"2": "Utricular"}, inplace=True)
	df.rename(columns={"3": "Saggital"}, inplace=True)

	vals, names, xs = [],[],[]
	for i, col in enumerate(df.columns):
		vals.append(df[col].values)
		names.append(col)
		xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

	plt.boxplot(vals, labels=names)
	palette = ['r', 'g', 'b']
	for x, val, c in zip(xs, vals, palette):
		plt.scatter(x, val, alpha=0.4, color=c)



	plt.title('Interoperator DICE similarity')
	plt.ylabel('DICE similarity')
	plt.show()
	
