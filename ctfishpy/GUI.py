import napari
from napari.layers import Image, Layer
from napari.types import ImageData
import numpy as np 
from magicgui import magicgui
from copy import deepcopy
import cv2
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget


def find_tubes(	ct, 
				slice_to_detect = 0,
				dp = 1.3,
				pad = 0,
				min_distance = 180,
				min_rad = 0,
				max_rad = 150,
				min_thresh = 20,
				max_thresh = 80,
				):

	"""
	This wont work on 16 bit
	"""

	output = deepcopy(ct) # copy stack to label later

	# Convert slice_to_detect to gray scale and threshold
	ct_slice_to_detect = ct[slice_to_detect]
	ret, ct_slice_to_detect = cv2.threshold(ct_slice_to_detect, min_thresh, max_thresh, 
		cv2.THRESH_BINARY)

	ct_slice_to_detect = cv2.cvtColor(ct_slice_to_detect, cv2.COLOR_RGB2GRAY)

	if not ret: raise Exception('Threshold failed')

	# detect circles in designated slice
	circles = cv2.HoughCircles(ct_slice_to_detect, cv2.HOUGH_GRADIENT, dp=dp, 
				minDist = min_distance, minRadius = min_rad, maxRadius = max_rad) #param1=50, param2=30,
				
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
	auto_call=True,
	dp={"widget_type": "FloatSlider", 'min' : 100, 'max' : 200},
	pad={"widget_type": "Slider", 'min' : 0, 'max' : 20},
	min_distance={"widget_type": "Slider", 'min' : 0, 'max' : 300},
	min_rad={"widget_type": "Slider", 'min' : 0, 'max' : 100},
	max_rad={"widget_type": "Slider", 'min' : 101, 'max' : 300},
	min_thresh={"widget_type": "Slider", 'min' : 1, 'max' : 100},
	max_thresh={"widget_type": "Slider", 'min' : 101, 'max' : 255},
	layout='vertical',)
def tubeDetector(layer:Layer,
					dp:float=150,
					pad:int=0,
					min_distance:int = 180,
					min_rad:int = 0,
					max_rad:int = 150,
					min_thresh:int = 20,
					max_thresh:int = 101,
					) -> Layer:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		# print(dp,pad,min_distance,min_rad,max_rad,min_thresh,max_thresh)

		array = layer.metadata['og'] # get original scan
		_slice = int(layer.position[0]) # get the slice youre looking at

		circle_dict = find_tubes(array,
								slice_to_detect=_slice,
								dp=dp/100,
								pad=pad,
								min_distance=min_distance,
								min_rad=min_rad,
								max_rad=max_rad,
								min_thresh=min_thresh,
								max_thresh=max_thresh,
								)
		if circle_dict:
			labelled = circle_dict['labelled_stack']
			layer.metadata['circle_dict'] = circle_dict
			layer.data = labelled
	
	return

def inside_circle( x, y, a, b, r):
	#check if point x, y is in circle with centre and radius a, b ,r 
	return (x - a)**2 + (y - b)**2 < r**2

def assign_circle_order(pos, circles, ordered_circles):
	#check if the click is in a circle 
	#add to ordered_circles
	x, y = pos
	for (a, b, r) in circles:
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
	layout='vertical',)
def orderLabeller(layer:Layer, undo:bool) -> Layer:
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

	return
    

# class OrderLabellerClass(QWidget):
# 	def __init__(self, parent) -> None:
# 		super().__init__(parent,)

# 		self.undo_btn = QPushButton("Undo", self)
# 		self.done_btn = QPushButton("Done", self)
# 		self.ordered_circles

# 		layout = QVBoxLayout(self)
# 		layout.addWidget(self.undo_btn)
# 		layout.addWidget(self.done_btn)


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
	layout='horizontal',)
def spinner(layer:Layer, angle:int=0, reset_center:bool=False) -> Layer:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		original = layer.metadata['og']
		center_rotation = layer.metadata['center_rotation']
		layer.metadata['angle'] = angle

		if center_rotation == None:
			center_rotation = [int(original.shape[0]/2), int(original.shape[1]/2)]
		
		new_image = deepcopy(original)
		position = center_rotation[::-1] # flip because qt :(
		cv2.circle(new_image, np.array(center_rotation, dtype='uint16')[::-1], 5, color=(0,0,255), thickness=2)
		new_image = rotate_image(new_image, angle, is_label=False, center=position)
		layer.data = new_image
	
	return

def create_spinner(viewer, layer) -> None:
	widget = spinner

	viewer.window.add_dock_widget(widget, name="spinner", area='bottom')
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 1: # if left click
			layer.metadata['center_rotation'] = event.position[::-1] #flip because qt :(
			widget.update()
		return

	@spinner.reset_center.clicked.connect
	def reset_spinner():
		layer.metadata['center_rotation'] = None
		widget.update()
		return

	return

# TODO add q shortcut

@magicgui(
	auto_call=True,
	reset={"widget_type": "PushButton"},
	layout='vertical',)
def fishRuler(layer:Layer, reset:bool=False) -> Layer:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!
		if 'head' in layer.metadata.keys():
			original = layer.metadata['og']
			head = layer.metadata['head']
			tail = layer.metadata['tail']
			font = cv2.FONT_HERSHEY_SIMPLEX

			new_image = deepcopy(original)
			if head!=0: # this doesnt work with none
				head_pos = np.array(head)
				x, y = head_pos
				cv2.circle(new_image, np.array(head_pos, dtype='uint16'), 5, color=(255,0,0), thickness=2)
				cv2.putText(img=new_image, text='head', 
									org=np.array(head_pos, dtype='uint16'), fontFace=font, 
									fontScale=0.5, color=(255,255,255), thickness=1)			
			if tail!=0:
				tail_pos = np.array(tail)
				x, y = tail_pos
				cv2.circle(new_image, np.array(tail_pos, dtype='uint16'), 5, color=(0,255,0), thickness=1)
				cv2.putText(img=new_image, text='tail', 
									org=np.array(tail_pos, dtype='uint16'), fontFace=font, 
									fontScale=0.5, color=(255,255,255), thickness=1)	
			if head!=0 and tail!=0:
				cv2.line(new_image, pt1=np.array(head_pos, dtype='uint16'),
									pt2=np.array(tail_pos, dtype='uint16'), 
									color=(255,255,255), thickness=1)
			
			layer.data = new_image
	return

def create_fishRuler(viewer, layer) -> None:
	widget = fishRuler

	viewer.window.add_dock_widget(widget, name="Measure length")
	viewer.layers.events.changed.connect(widget.reset_choices)

	@layer.mouse_drag_callbacks.append
	def get_event(layer, event):
		if event.button == 1: # if left click
			layer.metadata['head'] = event.position[::-1]
			widget.update()
		if event.button == 2: # if left click
			layer.metadata['tail'] = event.position[::-1]
			widget.update()
		return

	@fishRuler.reset.clicked.connect
	def reset_ruler():
		layer.metadata['head'] = 0
		layer.metadata['tail'] = 0
		widget.update()
		return

	return