import ctfishpy
import numpy as np

ctreader = ctfishpy.CTreader()

# dataset = 'Zac'
# sample = [40,256,421,423,242,582,257,443,461,527]
dataset = 'Mariel'
sample = [40,256,421,423,242,589,463,259,459,530]
# dataset = 'common'
# sample = [40,256,421,423,242]

for n in sample:
	ct, metadata = ctreader.read(n, align=True)
	ct = ((ct - ct.min()) / (ct.ptp() / 255.0)).astype(np.uint8) 
	ctreader.write_scan(dataset, ct, n, compression = 4, dtype='uint8')




