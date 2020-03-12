from ctfishpy.Lumpfish import Lumpfish
from ctfishpy.CTreader import CTreader
import gc

gc.collect()

CTreader = CTreader()
lump = Lumpfish()

ignore = [27, 29, 43, 47, 54, 56, 57, 59, 60, 62]

'''
27 hires
29 didn't work
43 can't tell
56, 57, 60, not constructed
rest: ¯|_(ツ)_/¯
'''

for i in range(0,64):
	if i in ignore: continue
	print('\n', i, '\n')
	ct, stack_metadata = lump.read_tiff(i, r = (0,100), scale = 100)

	crop_data = lump.readCrop(i)
	scale = [crop_data['scale'], stack_metadata['scale']]
	cropped_ordered_cts = lump.crop(ct, crop_data['ordered_circles'], scale = scale)

	lump.write_clean(i, cropped_ordered_cts, stack_metadata)
	ct, stack_metadata = None, None
	crop_data, cropped_ordered_cts = None, None
	gc.collect()
