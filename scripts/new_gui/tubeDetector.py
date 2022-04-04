from cv2 import circle
from importlib_metadata import metadata
import ctfishpy

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Layer
from napari.types import ImageData
from napari import Viewer
import cv2

@magicgui(
	# call_button='Detect',
	auto_call=True,
	dp={"widget_type": "FloatSlider", 'min' : 100, 'max' : 200},
	pad={"widget_type": "Slider", 'min' : 0, 'max' : 20},
	finished={"widget_type": "PushButton"},
	layout='vertical',)
def tubeDetector(layer:Layer, dp:float, pad:int, finished:bool=False) -> Layer:
	if layer is not None and finished==False:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		
		lump = ctfishpy.Lumpfish()

		array = layer.metadata['og'] # get original scan
		_slice = int(layer.position[0]) # get the slice youre looking at

		circle_dict = lump.find_tubes(array, dp=dp/100, slice_to_detect=_slice, pad=pad)
		if circle_dict: 
			labelled = circle_dict['labelled_stack']
			layer.data = labelled
		return

@tubeDetector.called.connect
def notify():
	print('DETECTING!!!')

def tubeLabeller(layer:Layer, ):
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!




if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()

	path = '/home/wahab/Data/Local/EK_208_215'
	scan = lump.read_tiff(path, r=(100,110))

	scan = ctreader.to8bit(scan)
	scan = np.array([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in scan])
	m = {'og': scan}

	viewer = napari.Viewer()
	layer = viewer.add_image(scan, metadata=m)

	@layer.mouse_drag_callbacks.append
	def get_event(viewer, event):
		print(event.__dict__)
		if event.button == 1: # if left click
			print('CLICK')

	viewer.window.add_dock_widget(tubeDetector)
	viewer.layers.events.changed.connect(tubeDetector.reset_choices)

	napari.run()




