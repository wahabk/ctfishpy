from CTFishPy.CTreader import CTreader
import gc

gc.collect()

CTreader = CTreader()

ignore = [27, 29, 43, 47, 54, 56, 57, 59, 60, 62]

for i in range(0,64):
	if i in ignore: continue
	print('\n', i, '\n')
	ct, stack_metadata = CTreader.read_dirty(i, r = None, scale = 80)

	crop_data = CTreader.readCrop(i)
	scale = [crop_data['scale'], stack_metadata['scale']]
	cropped_ordered_cts = CTreader.crop(ct, crop_data['ordered_circles'], scale = scale)

	CTreader.write_clean(i, cropped_ordered_cts, stack_metadata)
	ct, stack_metadata = None, None
	crop_data, cropped_ordered_cts = None, None
	gc.collect()
