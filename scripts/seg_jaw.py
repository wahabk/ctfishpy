import ctfishpy
import napari
import numpy as np

from napari.layers import Image, Layer, Labels
from magicgui import magicgui
from scipy.ndimage import zoom

from copy import deepcopy
import cv2

from skimage.segmentation import flood

@magicgui(auto_call=True,
	threshold={"widget_type": "Slider", "max":255, "min":0},
	new_value={"widget_type": "SpinBox", "max":255, "min":0},
	only_in={"widget_type": "CheckBox"},
	seek={"widget_type": "CheckBox"},
	# reset_center={"widget_type": "PushButton"},
	layout='Horizontal',)
def labeller(layer:Layer, label_layer:Labels, threshold:int=125, new_value:int=1, only_in:bool=False, seek=False) -> None: # reset_center:bool=False
	if layer is not None:
		if label_layer is not None:
			assert isinstance(layer.data, np.ndarray)  # it will be!
			assert isinstance(label_layer.data, np.ndarray)  # it will be!

			print(layer.metadata)

			label = deepcopy(label_layer.data)
			image = layer.data

			point = layer.metadata['point']
			if point is not None:
				if len(point) == 3:
					point = tuple([int(x) for x in point])
					new_label = None
					new_label = flood(image, point, tolerance=threshold)

					print(new_label.min(), new_label.max(), new_label.shape)
					print(label.min(), label.max(), label.shape)

					label_layer.data[new_label==True] = new_value
	
	if seek == False:
		layer.metadata['point'] = None
	return 

def create_labeller(viewer, layer) -> None:
	widget = labeller
	layer.metadata['point'] = None

	viewer.window.add_dock_widget(widget, name="labeller", area='right')
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 2: # if left click
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
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	ctreader = ctfishpy.CTreader(dataset_path)

	scan = ctreader.read(1)

	scan = ctreader.to8bit(scan)
	scan = scan[1000:]
	# scan = zoom(scan, 0.5)

	# point = (1368, 328, 356)

	# temp_label = flood_fill(scan, seed_point=point, new_value=1)

	# ctreader.view(scan, temp_label)

	# ctreader.view(scan)
	print(scan.shape)
	lab = label(scan)

	print(lab.min(), lab.max(), lab.shape)

	#TODO rewrite this one first as qt widget
	#TODO write only in
	#TODO write sweep thresh