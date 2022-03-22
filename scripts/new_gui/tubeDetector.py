from cv2 import circle
from importlib_metadata import metadata
import ctfishpy

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Layer
from napari.types import ImageData
import cv2

@magicgui(
	# call_button='Detect',
	auto_call=True,
	dp={"widget_type": "FloatSlider", 'min' : 100, 'max' : 200},
	pad={"widget_type": "Slider", 'min' : 0, 'max' : 20},
	finished={"widget_type": "CheckBox"},
	layout='vertical',)
def tubeDetector(layer:Layer, dp:float, pad:int, finished:bool,) -> Layer:
	if layer is not None:
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

if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()

	path = '/home/wahab/Data/Local/EK_208_215'
	scan = lump.read_tiff(path, r=(100,110))

	scan = ctreader.to8bit(scan)
	scan = np.array([cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) for s in scan])
	m = {'og': scan}

	viewer = napari.Viewer()
	viewer.add_image(scan, metadata=m)
	viewer.window.add_dock_widget(tubeDetector)
	viewer.layers.events.changed.connect(tubeDetector.reset_choices)

	napari.run()




