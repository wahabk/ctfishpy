from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
from .. CTreader import CTreader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

class MainWindow(QMainWindow):

	def __init__(self, stack):
		super().__init__()
		self.npstack = stack
		self.initUI()

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		viewer = Viewer(self.npstack)
		self.setCentralWidget(viewer)
		#widget.findChildren(QWidget)[0]

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, viewer.width(), viewer.height())
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

#https://stackoverflow.com/questions/34644808/set-vertical-alignment-of-qformlayout-qlabel

class Viewer(QWidget):

	def __init__(self, stack, stride = 1):
		super().__init__()
		self.npstack = stack
		self.slice = 0
		self.ogstack = stack
		self.label = QLabel(self)
		self.dp = 1.3

		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.stack_size = stack.shape[0]-1
		self.stride = stride

		#check length of image shape to check if image is grayscale or color
		if len(stack.shape) == 3: self.grayscale = True
		elif len(stack.shape) == 4: self.grayscale = False
		else: raise ValueError('[viewer] Cant tell if stack is color or grey scale')
		self.initSlider()
		self.initDetector()
		self.initUI()


	def initUI(self):
		#initialise UI
		self.update()
		self.slider.setGeometry(10, self.pixmap.height()+10, self.pixmap.width(), 20)
		self.detector.setGeometry(10, self.pixmap.height()+10+self.slider.height(), self.pixmap.width(), 20)
		self.label.setMargin(10)
		self.setGeometry(0, 0, self.pixmap.width()+20, self.pixmap.height()+20+self.slider.height()*2+self.detector.height()*2)
		self.slider.valueChanged.connect(self.updateSlider)
		self.detector.valueChanged.connect(self.updateDetector)

	def update(self):
		#Update displayed image
		self.image = self.np2qt(self.npstack[self.slice])
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
		#transform np cv2 image to qt format
		ctreader = CTreader()
		circle_dict  = ctreader.find_tubes(self.ogstack, dp = self.dp)
		if circle_dict: self.npstack = circle_dict['labelled_stack']
		else: self.npstack = self.ogstack

		if self.grayscale == True:
			height, width = image.shape
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

	def updateSlider(self):
		self.slice = self.slider.value()
		self.update()	

	def initDetector(self):
		self.detector = QSlider(Qt.Horizontal, self)
		self.detector.setMinimum(100)
		self.detector.setMaximum(200)
		self.detector.setSingleStep(1)

	def updateDetector(self):
		self.dp = self.detector.value()/100
		self.update()

def mainwin(stack):
	app = QApplication(sys.argv)
	win = MainWindow(stack)
	win.show()
	app.exec_()

