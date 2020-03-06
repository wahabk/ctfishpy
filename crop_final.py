from ctfishpy.GUI.circle_order_labeller import circle_order_labeller
from ctfishpy.GUI.tubeDetector import detectTubes
from ctfishpy.CTreader import CTreader
from ctfishpy.Lumpfish import Lumpfish

CTreader = CTreader()
lump = Lumpfish()

for i in range(58,64):
	print(i)
	ct, stack_metadata = lump.read_dirty(i, r = (0,10), scale = 40)
	circle_dict = detectTubes(ct)
	CTreader.view(ct)
	ordered_circles, numbered = circle_order_labeller(
		circle_dict['labelled_stack'], circle_dict['circles'])
	CTreader.view(numbered)
	lump.saveCrop(n = i, 
		ordered_circles = ordered_circles, 
		metadata = stack_metadata)
