from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
import numpy as np
import cv2
import sys
from copy import deepcopy

class mainView(QMainWindow):

	def __init__(self, stack, label = None, thresh = False):
		super().__init__()
		self.thresh = thresh
		self.label = label
		self.stack_length = stack.shape[0]
		# convert 16 bit grayscale to 8 bit
		# by mapping the data range to 0 - 255
		if stack.dtype == 'uint16':
			self.stack = ((stack - stack.min()) / (stack.ptp() / 255.0)).astype(np.uint8) 
		else:
			self.stack = stack
		self.initUI()

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		self.viewer = Viewer(self.stack, label = self.label, parent = self, thresh = self.thresh)
		self.setCentralWidget(self.viewer)
		#widget.findChildren(QWidget)[0]
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, self.viewer.width(), self.viewer.height())
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

	def updateStatusBar(self, slice_ = 0):
		self.statusBar().showMessage(f'{slice_}/{self.stack_length}')


class Viewer(QWidget):

	def __init__(self, stack, label = None, thresh = False, stride = 1, parent = None):
		super().__init__()

		# init variables
		
		self.ogstack = deepcopy(stack)
		self.stack_size = stack.shape[0]
		self.stride = stride
		self.slice = 0
		self.parent = parent
		self.min_thresh = 0
		self.max_thresh = 250
		self.thresh = thresh
		self.label = deepcopy(label)
		self.pad = 20
		self.is_colored = False

		# label stack
		if self.label is not None:
			self.thresh == False
			# unpack images to convert each one to grayscale
			self.ogstack = np.array([np.stack((img,)*3, axis=-1) for img in self.ogstack])
			self.is_colored = True
			# change pixels to red if in label
			self.ogstack[label == 1, :] = [255, 0, 0]
			self.ogstack[label == 2, :] = [255, 255, 0]
			self.ogstack[label == 3, :] = [0, 0, 255]
			self.ogstack[label == 4, :] = [0, 255, 0]
			self.ogstack[label == 5, :] = [255, 0, 255]
		
		if np.max(stack) < 10: #fix labels so you can view them if are main stack
			print('LABELLING')
			label = deepcopy(stack)
			temp = np.zeros(stack.shape)
			temp = np.array([np.stack((img,)*3, axis=-1) for img in temp])
			self.is_colored = True
			temp[label == 1, :] = [255, 0, 0]
			temp[label == 2, :] = [255, 255, 0]
			temp[label == 3, :] = [0, 0, 255]
			temp[label == 4, :] = [0, 255, 0]
			self.ogstack = temp
			print(f'SHAPE{self.ogstack.shape}, {self.ogstack.max()}')

		#if self.ogstack.shape[0] == self.ogstack.shape[1]: self.is_single_image = True
		if len(self.ogstack.shape) == 2: self.is_single_image = True
		else: self.is_single_image = False

		scale = 200
		im = self.ogstack[0]
		height = int(im.shape[0] * scale / 100)
		width = int(im.shape[1] * scale / 100)
		dim = (width, height)

		self.ogstack = np.array(self.ogstack)
		# resize image
		resized = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in self.ogstack]
		self.ogstack = np.array(resized)
		print(f'SHAPE{self.ogstack.shape}, {self.ogstack.max()}')

		# set background colour to cyan
		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.qlabel = QLabel(self)
		self.qlabel.setMargin(10)
		self.initUI()

	def initUI(self):
		pad = self.pad
		self.update()
		if self.is_single_image == False:
			self.initSlider()
			self.slider.valueChanged.connect(self.updateSlider)
			self.setGeometry(0, 0, 
			self.pixmap.width() + pad, 
			self.pixmap.height() + pad + self.slider.height()*2 + pad)

		elif self.is_single_image: 
			self.setGeometry(0, 0, 
			self.pixmap.width() + pad, 
			self.pixmap.height() + pad*2)

		if self.thresh == True: 
			self.initThresholdSliders()
			self.setGeometry(0, 0, 
				self.pixmap.width() + pad, 
				self.pixmap.height()+ pad + self.slider.height()*2 + self.min_thresh_slider.height()*2 + self.max_thresh_slider.height()*2)
			self.min_thresh_slider.valueChanged.connect(self.update_min_thresh)
			self.max_thresh_slider.valueChanged.connect(self.update_max_thresh)


	def update(self):

		if self.slice > self.stack_size-1: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size-1
		
		if self.thresh == True:
			ret, self.image  = cv2.threshold(self.ogstack[self.slice], 
				self.min_thresh, self.max_thresh, cv2.THRESH_BINARY)
		elif self.is_single_image == True: self.image = self.ogstack
		else: self.image = self.ogstack[self.slice]

		# transform image to qimage and set pixmap
		self.image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.qlabel.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		if self.is_single_image: return
		#scroll through slices and go to beginning if reached max
		self.slice = self.slice + int(event.angleDelta().y()/120)*self.stride
		if self.slice > self.stack_size-1: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size-1
		self.slider.setValue(self.slice)
		self.update()

	def np2qt(self, image):
		# transform np cv2 image to qt format

		# check length of image shape to check if image is grayscale or color
		if len(image.shape) == 2: grayscale = True
		elif len(image.shape) == 3: grayscale = False
		else: raise ValueError('[Viewer] Cant tell if stack is color or grayscale, weird shape :/')
		if self.is_single_image: grayscale = True

		# convert npimage to qimage depending on color mode
		if grayscale == True:
			height, width = self.image.shape
			bytesPerLine = width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		else:
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

	def initSlider(self):
		self.slider = QSlider(Qt.Horizontal, self)
		self.slider.setMinimum(0)
		self.slider.setMaximum(self.stack_size)
		self.slider.setGeometry(10, self.pixmap.height()+10, self.pixmap.width(), 20)

	def updateSlider(self):
		self.slice = self.slider.value()
		self.update()
		self.parent.updateStatusBar(self.slice)

	def initThresholdSliders(self):
		self.min_thresh_slider = QSlider(Qt.Horizontal, self)
		self.min_thresh_slider.setMinimum(0) # betweeen 100 and 200
		self.min_thresh_slider.setMaximum(150) # -for decimal places
		self.min_thresh_slider.setSingleStep(1)
		self.min_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height(), self.pixmap.width(), 20)
		
		self.max_thresh_slider = QSlider(Qt.Horizontal, self)
		self.max_thresh_slider.setMinimum(100)
		self.max_thresh_slider.setMaximum(255)
		self.max_thresh_slider.setSingleStep(1)
		self.max_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height()+self.min_thresh_slider.height(), self.pixmap.width(), 20)

	def update_min_thresh(self):
		self.min_thresh = self.min_thresh_slider.value() # divide by 100 to get decimal points
		self.update()

	def update_max_thresh(self):
		self.max_thresh = self.max_thresh_slider.value() # divide by 100 to get decimal points
		self.update()


def mainViewer(stack, label, thresh):
	app = QApplication(sys.argv)
	win = mainView(stack, label, thresh)
	win.show()
	app.exec_()
	return
