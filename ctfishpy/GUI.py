import napari
from napari.layers import Image, Layer
from tqdm import tqdm
import tifffile as tiff
import pandas as pd
import numpy as np 
from magicgui import magicgui
from .Lumpfish import Lumpfish


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
		
		lump = Lumpfish()

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


    