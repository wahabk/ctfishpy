from ctfishpy
import gc

CTreader = ctfishpy.CTreader()
lump = ctfishpy.Lumpfish()

skip = [27, 54]

for i in range(0,64):
	if i in skip: continue
	print('\n', i, '\n')
	ct, stack_metadata = lump.read_tiff(i, r = None, scale = 100)

	crop_data = lump.readCrop(i)
	scale = [crop_data['scale'], stack_metadata['scale']]
	cropped_ordered_cts = lump.crop(ct, crop_data['ordered_circles'], scale = scale)

	lump.write_clean(i, cropped_ordered_cts, stack_metadata)
	ct, stack_metadata = None, None
	crop_data, cropped_ordered_cts = None, None
	gc.collect()
