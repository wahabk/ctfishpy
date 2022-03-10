import ctfishpy

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image
from napari.types import ImageData
import cv2

def rotate_array(array, angle, is_label, center=None):
	new_array = []
	for a in array:
		a_rotated = rotate_image(a, angle=angle, is_label=is_label, center=center)
		new_array.append(a_rotated)
	return np.array(new_array)

def rotate_image(image, angle, is_label, center=None):
	"""
	Rotate images properly using cv2.warpAffine
	since it provides more control eg over center

	parameters
	image : 2d np array
	angle : angle to spin
	center : provide center if you dont want to spin around true center
	"""
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	if center:
		image_center = center
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	# THIS HAS TO BE NEAREST NEIGHBOUR BECAUSE LABELS ARE CATEGORICAL
	if is_label:
		interpolation_flag = cv2.INTER_NEAREST
	else:
		interpolation_flag = cv2.INTER_LINEAR
	result = cv2.warpAffine(
		image, rot_mat, image.shape[1::-1], flags=interpolation_flag
	)
	return result

@magicgui(
	angle={"widget_type": "Slider", 'max': 360, 'min':0},
	layout='vertical',)
def spin_scan(layer:Image, angle:int=0):
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!

		array = layer.data

		new_array = rotate_array(array, angle, is_label=False, center=None)
		layer.data = new_array

		return

# @spin_scan.called.connect
# def print_mean(value):
#     """Callback function that accepts an event"""
#     # the value attribute has the result of calling the function
#     print(f'spun fish with angle {value}')


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()

	fish , metadata = ctreader.read(40)

	viewer = napari.Viewer()
	viewer.add_image(fish)
	viewer.window.add_dock_widget(spin_scan)

	napari.run()
	

