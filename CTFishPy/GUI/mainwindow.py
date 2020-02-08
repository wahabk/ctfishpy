from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider
from qtpy.QtGui import QFont, QPixmap, QImage
from qtpy.QtCore import Qt, QTimer
from .. import CTreader
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

		layout = QVBoxLayout()
		layout.addWidget(Viewer(self.npstack))
		layout.addWidget(Slider())
		
		widget = QWidget()
		widget.setLayout(layout)
		self.setCentralWidget(widget)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		#self.resize(viewer.pixmap.width(), viewer.pixmap.height()+200)
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

#https://stackoverflow.com/questions/34644808/set-vertical-alignment-of-qformlayout-qlabel

class Viewer(QWidget):

	def __init__(self, stack, skip = 10):
		super().__init__()
		self.npstack = stack
		self.slice = 0
		self.label = QLabel(self)
		self.stack_size = stack.shape[0]-1
		self.skip = skip

		#check length of image shape to check if image is grayscale or color
		if len(stack.shape) == 3: self.grayscale = True
		elif len(stack.shape) == 4: self.grayscale = False
		else: raise ValueError('[viewer] Cant tell if stack is color or grey scale')

		self.initUI()

	def initUI(self):
		#initialise UI
		self.update()
		self.resize(self.pixmap.width(), self.pixmap.height())

	def update(self):
		#Update displayed image
		self.image = self.np2qt(self.npstack[self.slice])
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.label.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		#scroll through slices and go to beginning if reached max
		self.slice = self.slice + int(event.angleDelta().y()/120)*self.skip
		if self.slice > self.stack_size: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size
		self.update()

	def np2qt(self, image):
		#transform np cv2 image to qt format
		if self.grayscale == True:
			height, width = image.shape
			bytesPerLine = width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		else:
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

class Slider(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		sld = QSlider(Qt.Horizontal, self)
		#sld.valueChanged.connect(lcd.display)

		self.setGeometry(100, 100, 0, 0)


def mainwin(stack):
	app = QApplication(sys.argv)
	win = MainWindow(stack)
	win.show()
	app.exec_()