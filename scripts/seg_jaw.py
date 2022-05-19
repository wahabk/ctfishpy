import ctfishpy
import napari
import numpy as np

from napari.layers import Image, Layer, Labels
from magicgui import magicgui

from copy import deepcopy
import cv2

from skimage.segmentation import flood_fill

@magicgui(auto_call=True,
	threshold={"widget_type": "Slider", "max":255, "min":0},
	# reset_center={"widget_type": "PushButton"},
	layout='vertical',)
def spinner(layer:Layer, labels:Labels, threshold:int=10) -> None: # reset_center:bool=False
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		assert isinstance(Labels.data, np.ndarray)  # it will be!

		point = layer.metadata['point']
		if point:
			new_label = flood_fill(layer.data, point, new_value=1, tolerance=threshold)
			labels.data = new_label
	
	return

def create_spinner(viewer, layer) -> None:
	widget = spinner
	layer.metadata['point'] = None

	viewer.window.add_dock_widget(widget, name="spinner", area='bottom')
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 1: # if left click
			layer.metadata['point'] = event.position[::-1] #flip because qt :(
			widget.update()
		return

	# @spinner.reset_center.clicked.connect
	# def reset_spinner():
	# 	layer.metadata['center_rotation'] = None
	# 	widget.update()
	# 	return

	return

def flood(scan):
	viewer = napari.Viewer()
	layer = viewer.add_image(scan)
	viewer.show()


if __name__ == "__main__":

	ctreader = ctfishpy.CTreader()

	scan = ctreader.read(40)

	# ctreader.view(scan)
	flood(scan)

