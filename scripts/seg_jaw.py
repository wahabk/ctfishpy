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
def labeller(layer:Layer, labels:Labels, threshold:int=125) -> None: # reset_center:bool=False
	if layer is not None:
		if labels is not None:
			print(layer)
			print(labels)
			assert isinstance(layer.data, np.ndarray)  # it will be!
			assert isinstance(labels.data, np.ndarray)  # it will be!

			print(layer.metadata)

			point = layer.metadata['point']
			if point is not None:
				if len(point) == 3:
					point = tuple([int(x) for x in point])
					print(f"clicked {point}, threshold {threshold}")
					new_label = flood_fill(layer.data, point, new_value=1, tolerance=threshold, in_place=False)


					print(new_label.min(), new_label.max(), new_label.shape)

					labels.data = new_label
	
	return 

def create_labeller(viewer, layer) -> None:
	widget = labeller
	layer.metadata['point'] = None

	viewer.window.add_dock_widget(widget, name="labeller", area='bottom')
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 1: # if left click
			layer.metadata['point'] = event.position #flip because qt :(
			widget.update()
		return

	# @spinner.reset_center.clicked.connect
	# def reset_spinner():
	# 	layer.metadata['center_rotation'] = None
	# 	widget.update()
	# 	return

	return

def label(scan):
	l = np.zeros_like(scan)

	viewer = napari.Viewer()
	layer = viewer.add_image(scan)
	label_layer = viewer.add_labels(l)

	create_labeller(viewer, layer)

	viewer.show(block=True)

	return label_layer.data


if __name__ == "__main__":

	ctreader = ctfishpy.CTreader()

	scan = ctreader.read(1)

	scan = ctreader.to8bit(scan)

	point = (1368, 328, 356)

	temp_label = flood_fill(scan, seed_point=point, new_value=1)

	ctreader.view(scan, temp_label)

	# ctreader.view(scan)
	lab = label(scan)

	print(lab.min(), lab.max(), lab.shape)

