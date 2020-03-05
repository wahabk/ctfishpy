from CTFishPy.GUI.circle_order_labeller import circle_order_labeller
from CTFishPy.GUI.tubeDetector import detectTubes
from CTFishPy.CTreader import CTreader

CTreader = CTreader()

for i in range(58,64):
	print(i)
	ct, stack_metadata = CTreader.read_dirty(i, r = None, scale = 40)
	circle_dict = detectTubes(ct)
	CTreader.view(ct)
	ordered_circles, numbered = circle_order_labeller(
		circle_dict['labelled_stack'], circle_dict['circles'])
	CTreader.view(numbered)
	CTreader.saveCrop(n = i, 
		ordered_circles = ordered_circles, 
		metadata = stack_metadata)
