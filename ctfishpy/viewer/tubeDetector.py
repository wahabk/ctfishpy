from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
from ..controller import Lumpfish
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def to8bit(img):
	"""
	Change img from 16bit to 8bit by mapping the data range to 0 - 255
	"""
	if img.dtype == "uint16":
		new_img = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8)
		return new_img
	else:
		print("image already 8 bit!")
		return img

def to8bit_stack(stack):
	return np.array([to8bit(i) for i in stack], dtype='uint8')

class detectCircles(QMainWindow):

	def __init__(self, stack):
		super().__init__()
		self.stack = stack
		self.viewer = Viewer(self.stack, parent = self)
		self.initUI()

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		self.setCentralWidget(self.viewer)
		#widget.findChildren(QWidget)[0]

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, self.viewer.width(), self.viewer.height())
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
			self.close()


class Viewer(QWidget):

	def __init__(self, stack, stride = 10, parent = None):
		super().__init__()
		#init cariables
		self.npstack = to8bit_stack(stack)
		self.ogstack = to8bit_stack(stack)
		self.stack_size = stack.shape[0]-1
		self.stride = stride
		self.slice = 0
		self.pad = 0
		self.dp = 1.3
		self.circle_dict = []
		self.locked = False
		self.parent = parent

		#set background colour to cyan
		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.initUI()

	def initUI(self):
		self.label = QLabel(self)
		self.label.setMargin(10)
		self.update()
		self.initSliders()
		self.setGeometry(0, 0, self.pixmap.width()+20, self.pixmap.height()+20+self.slider.height()*2+self.detector.height()*2+self.padder.height()*2)
		
		self.slider.valueChanged.connect(self.updateSlider)
		self.detector.valueChanged.connect(self.updateDetector)
		self.padder.valueChanged.connect(self.updatePadder)

		self.b1 = QPushButton("Lock slice", self)
		self.b1.setCheckable(True)
		self.b1.toggle()
		self.b1.clicked.connect(self.lockSlice)
		self.b1.move(30, 50)

		b2 = QPushButton("Next step", self)
		b2.move(150, 50)
		b2.clicked.connect(self.Next)

	def lockSlice(self):
		if self.b1.isChecked():
			self.locked = False
			self.b1.setText('Slice Locked')
		else:
			self.locked = True
			self.b1.setText('Lock slice')

	def update(self):
		# Update displayed image
		# detect circles if unloacked
		if self.locked == False:
			lump = Lumpfish()
			self.circle_dict  = lump.find_tubes(self.ogstack, dp = self.dp, slice_to_detect = self.slice, pad = self.pad)
			if self.circle_dict: self.npstack = self.circle_dict['labelled_stack']
			else: self.npstack = self.ogstack
		
		# transform image to qimage and set pixmap
		self.image = self.npstack[self.slice]
		self.image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.label.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		#scroll through slices and go to beginning if reached max
		self.slice = self.slice + int(event.angleDelta().y()/120)*self.stride
		if self.slice > self.stack_size: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size
		self.slider.setValue(self.slice)
		self.update()

	def np2qt(self, image):
		# transform np cv2 image to qt format

		# check length of image shape to check if image is grayscale or color
		if len(self.npstack.shape) == 3: grayscale = True
		elif len(self.npstack.shape) == 4: grayscale = False
		else: raise ValueError('[viewer] Cant tell if stack is color or grey scale')

		# convert npimage to qimage
		if grayscale == True:
			height, width = self.image.shape
			bytesPerLine = width
			return QImage(self.image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		else:
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			return QImage(self.image.data, width, height, bytesPerLine, QImage.Format_RGB888)

	def initSliders(self):
		self.slider = QSlider(Qt.Horizontal, self)
		self.slider.setMinimum(0)
		self.slider.setMaximum(self.stack_size)
		self.slider.setGeometry(10, self.pixmap.height()+10, self.pixmap.width(), 20)

		self.detector = QSlider(Qt.Horizontal, self)
		self.detector.setMinimum(100) # betweeen 100 and 200
		self.detector.setMaximum(200) # -for decimal places
		self.detector.setSingleStep(1)
		self.detector.setGeometry(10, self.pixmap.height()+10+self.slider.height(), self.pixmap.width(), 20)
		
		self.padder = QSlider(Qt.Horizontal, self)
		self.padder.setMinimum(1)
		self.padder.setMaximum(100) 
		self.padder.setSingleStep(1)
		self.padder.setGeometry(10, self.pixmap.height()+10+self.slider.height()+self.detector.height(), self.pixmap.width(), 20)

	def updateSlider(self):
		self.slice = self.slider.value()
		self.update()	

	def updateDetector(self):
		self.dp = self.detector.value()/100 # divide by 100 to get decimal points
		self.update()

	def updatePadder(self):
		self.pad = self.padder.value() # divide by 100 to get decimal points
		self.update()

	def Next(self):
		self.parent.close()

def detectTubes(stack):
	app = QApplication(sys.argv)
	win = detectCircles(stack)
	win.show()
	app.exec_()
	return win.viewer.circle_dict

