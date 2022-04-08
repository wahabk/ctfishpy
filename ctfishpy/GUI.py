import napari
from napari.layers import Image, Layer
from napari.types import ImageData
import numpy as np 
from magicgui import magicgui
from copy import deepcopy
import cv2
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget


def find_tubes(ct, minDistance = 180, minRad = 0, maxRad = 150, 
	thresh = [20, 80], slice_to_detect = 0, dp = 1.3, pad = 0):

	"""
	This wont work on 16 bit
	"""

	# Find fish tubes
	# output = ct.copy() # copy stack to label later
	output = deepcopy(ct)

	# Convert slice_to_detect to gray scale and threshold
	# ct_slice_to_detect = cv2.cvtColor(ct[slice_to_detect], cv2.COLOR_BGR2GRAY)
	ct_slice_to_detect = ct[slice_to_detect]
	min_thresh, max_thresh = thresh
	ret, ct_slice_to_detect = cv2.threshold(ct_slice_to_detect, min_thresh, max_thresh, 
		cv2.THRESH_BINARY)

	ct_slice_to_detect = cv2.cvtColor(ct_slice_to_detect, cv2.COLOR_RGB2GRAY)

	if not ret: raise Exception('Threshold failed')

	# detect circles in designated slice
	circles = cv2.HoughCircles(ct_slice_to_detect, cv2.HOUGH_GRADIENT, dp=dp, 
				minDist = minDistance, minRadius = minRad, maxRadius = maxRad) #param1=50, param2=30,
				
	if circles is None: return
	else:
		# add pad value to radii

		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int") # round up
		circles[:,2] = circles[:,2] + pad

		# loop over the (x, y) coordinates and radius of the circles
		for i in output:
			for (x, y, r) in circles:
				# draw the circle in the output image, then draw a rectangle
				# corresponding to the center of the circle
				cv2.circle(i, (x, y), r, (0, 0, 255), 2)
				cv2.rectangle(i, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

		circle_dict  =  {'labelled_img'  : output[slice_to_detect],
							'labelled_stack': output, 
							'circles'     : circles}
		return circle_dict

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

		array = layer.metadata['og'] # get original scan
		_slice = int(layer.position[0]) # get the slice youre looking at

		circle_dict = find_tubes(array, dp=dp/100, slice_to_detect=_slice, pad=pad)
		if circle_dict: 
			labelled = circle_dict['labelled_stack']
			layer.metadata['circle_dict'] = circle_dict
			layer.data = labelled

		# TODO fix finished button by connecting in caller?
		if finished:
			viewer = napari.current_viewer()
			viewer.close()
		
		return

# @tubeDetector.finished.connect
# def notify():
# 	print('finished')

def inside_circle( x, y, a, b, r):
	#check if point x, y is in circle with centre and radius a, b ,r 
	return (x - a)**2 + (y - b)**2 < r**2

def assign_circle_order(pos, circles, ordered_circles):
	#check if the click is in a circle 
	#add to ordered_circles
	x, y = pos
	for (a, b, r) in circles:
		# print(x,y,a,b,r)
		if inside_circle(x, y, a, b, r):
			#check circle not already in order
			if len(ordered_circles)>0 and True in [inside_circle(x,y,i,j,k) for (i,j,k) in ordered_circles]:
				return ordered_circles
			ordered_circles.append([a, b, r])
			
	return ordered_circles

def number_scan(scan, order):
	new_scan = scan.copy()
	for num, (x, y, r) in enumerate(order):
		pos = (x,y)
		font = cv2.FONT_HERSHEY_SIMPLEX
		[cv2.putText(img=img, text=str(num+1), org=pos, fontFace=font, fontScale=4, 
								color=(255,255,255), thickness=2) for img in new_scan]
	return new_scan


@magicgui(
	# call_button='Detect',
	auto_call=True,
	undo={"widget_type": "PushButton"},
	finished={"widget_type": "PushButton"},
	layout='vertical',)
def orderLabeller(layer:Layer, undo:bool, finished:bool):
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		m = layer.metadata
		if 'pos' in dir(orderLabeller) and isinstance(orderLabeller.pos, tuple): # if not clicked yet
			pos = orderLabeller.pos[1:] # get rid of z
			pos = pos[::-1] # reverse because qt :(

			new_order = assign_circle_order(pos, m['circles'], m['ordered_circles'])
			orderLabeller.pos = False

			if new_order:
				m['ordered_circles'] = new_order # if new order change the metadata

		new_order = m['ordered_circles'] # if undoing read metadata
		scan = number_scan(m['og'], new_order)
		layer.data = scan
		return

def create_orderLabeller(viewer, layer) -> None:
	widget = orderLabeller

	viewer.window.add_dock_widget(widget, name="orderLabeller")
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 1: # if left click
			widget.pos = event.position
			widget.update()
		return

	@orderLabeller.undo.clicked.connect
	def undo():
		layer.metadata["ordered_circles"].pop() # remove last ordered circle
		orderLabeller.update()

	# @orderLabeller.finished.clicked.connect
	# def finished():
	# 	viewer.close()
    

# class OrderLabeller(QWidget):
# 	def __init__(self, parent) -> None:
# 		super().__init__(parent,)

# 		self.undo_btn = QPushButton("Undo", self)
# 		self.done_btn = QPushButton("Done", self)
# 		self.ordered_circles

# 		layout = QVBoxLayout(self)
# 		layout.addWidget(self.undo_btn)
# 		layout.addWidget(self.done_btn)

# 	def inside_circle(self, x, y, a, b, r):
# 		#check if point x, y is in circle with centre and radius a, b ,r 
# 		return (x - a)**2 + (y - b)**2 < r**2

# 	def assign_circle_order(self, pos, circles, ordered_circles):
# 		#check if the click is in a circle 
# 		#add to ordered_circles and deleted from original circles
# 		x, y = pos
# 		for (a, b, r) in circles:
# 			if self.inside_circle(x, y, a, b, r):
# 				#check circle not already in order
# 				if ordered_circles and True in [self.inside_circle(x,y,i,j,k) for (i,j,k) in ordered_circles]:
# 					break
# 				ordered_circles.append([a, b, r])

# 				image = self.number_image(image, len(ordered_circles), (a,b))
# 				break
# 		return ordered_circles

# 	def number_image(self, img, num, loc):

# 		font = cv2.FONT_HERSHEY_SIMPLEX
# 		return cv2.putText(img=img.copy(), text=str(num), org=loc, fontFace=font, fontScale=4, color=(255,255,255), thickness=2)

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

def rotate_array(array, angle, is_label, center=None):
	new_array = []
	for a in array:
		a_rotated = rotate_image(a, angle=angle, is_label=is_label, center=center)
		new_array.append(a_rotated)
	return np.array(new_array)

@magicgui(
	auto_call=True,
	angle={"widget_type": "Slider", 'max': 360, 'min':0},
	reset_center={"widget_type": "PushButton"},
	layout='vertical',)
def spinner(layer:Layer, angle:int=0, reset_center:bool=False) -> None:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		original = layer.metadata['og']
		center_rotation = layer.metadata['center_rotation']
		layer.metadata['angle'] = angle

		if center_rotation:
			position = center_rotation[::-1]
			new_image = rotate_image(original, angle, is_label=False, center=position)
			layer.data = new_image
		else:
			layer.data = rotate_image(original, angle, is_label=False, center=None)



def create_spinner(viewer, layer) -> None:
	widget = spinner

	viewer.window.add_dock_widget(widget, name="spinner")
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 1: # if left click
			layer.metadata['center_rotation'] = event.position
			widget.update()
		return

	@spinner.reset_center.clicked.connect
	def reset_spinner():
		layer.metadata['center_rotation'] = None
		widget.update()

# TODO find length
# TODO paint center of rotation in spinner
# TODO remove finished button and add q shortcut
