from ctfishpy.GUI.circle_order_labeller import circle_order_labeller
from ctfishpy.GUI.tubeDetector import detectTubes
import ctfishpy

CTreader = ctfishpy.CTreader()
lump = ctfishpy.Lumpfish()


for i in range(62,63):
	print(i)
	ct, stack_metadata = lump.read_dirty(i, r = None, scale = 40)
	circle_dict = detectTubes(ct)
	CTreader.view(ct)
	ordered_circles, numbered = circle_order_labeller(
		circle_dict['labelled_stack'], circle_dict['circles'])
	CTreader.view(numbered)
	lump.saveCrop(n = i, 
		ordered_circles = ordered_circles, 
		metadata = stack_metadata)
