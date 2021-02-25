import ctfishpy
import segmentation_models as sm
import numpy as np

def create_folds(sample, fold_size=3):
	if len(sample) % fold_size != 0 :
		raise Exception('folds dont fit in sample')
	folds = [sample[i:i+fold_size] for i in range(0,len(sample),fold_size)]
	return folds

wahab_samples 	= [78,200,240,277,330,337,341,462]
mariel_samples	= [421,423,242,463,259,459]
zac_samples		= [257,443]
# removing 527, 530, 582, 589 mutants = [527, 530, 582, 589]
sample = wahab_samples+mariel_samples
val_sample = [464,364,385,461] 


ctreader = ctfishpy.CTreader()
master = ctreader.mastersheet()
conditions = ['wt', 'het', 'hom']
#remove genotypes injected or mosaic etc
master = master[master['genotype'].isin(conditions)]
master = master[master['n'].isin(sample)]
master = master.sort_values(['genotype', 'age'])
master.to_csv('output/otoliths2segFINAL.csv')

# 40 218 removed from cv
# mcf2l = 341,337 
# 78 is odd 7 month
# removed olf het 464
young_wt = [385,423]
mid_wt_1 = [200,330]
mid_wt_2 = [364,277]
old_wt = [240,242]
young_het = [459,461]
mid_het = [257,259]
old_het = [462,463]
col_hom_1 = [421,443]
col_hom_2 = [582,589]
ncoa3_hom = [527,530]

folds = [young_het, mid_het, old_het, col_hom_1, col_hom_2, young_wt, mid_wt_1, mid_wt_2, old_wt, ncoa3_hom]
flat_list = [item for sublist in folds for item in sublist]

final_folds=[]
for i in range(len(folds)):
	val_sample = folds[i]
	train_sample = [x for x in flat_list if x not in val_sample]
	final_folds.append([i, train_sample, val_sample])

for n, train, val in final_folds:
	print('\n',n,train,val,'\n')
	unet = ctfishpy.Unet('Otoliths')
	unet.weightsname = 'CV_Weighted_Dice'
	unet.comment = 'CV_Weighted_Dice'
	unet.fold = n
	unet.train(train, val)
	unet.makeLossCurve()
	del unet

