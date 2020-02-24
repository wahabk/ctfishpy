from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
import numpy as np
import cv2
import sys

class mainView(QMainWindow):

	def __init__(self, stack):
		super().__init__()
		self.stack = stack
		self.initUI()

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		self.viewer = Viewer(self.stack, parent = self)
		self.setCentralWidget(self.viewer)
		#widget.findChildren(QWidget)[0]

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, self.viewer.width(), self.viewer.height())
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()


class Viewer(QWidget):

	def __init__(self, stack, stride = 10, parent = None):
		super().__init__()
		#init cariables
		self.ogstack = stack
		self.stack_size = stack.shape[0]-1
		self.stride = stride
		self.slice = 0
		self.parent = parent
		self.min_thresh = 50
		self.max_thresh = 150

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
		self.setGeometry(0, 0, self.pixmap.width()+20, self.pixmap.height()+20+self.slider.height()*2+self.min_thresh_slider.height()*2+self.max_thresh_slider.height()*2)
		
		self.slider.valueChanged.connect(self.updateSlider)
		self.min_thresh_slider.valueChanged.connect(self.update_min_thresh)
		self.max_thresh_slider.valueChanged.connect(self.update_max_thresh)

	def update(self):
		gray = cv2.cvtColor(self.ogstack[self.slice], cv2.COLOR_BGR2GRAY)
		ret, self.image  = cv2.threshold(gray, 
			self.min_thresh, self.max_thresh, cv2.THRESH_BINARY)
		
		# transform image to qimage and set pixmap
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
		if len(image.shape) == 2: grayscale = True
		elif len(image.shape) == 3: grayscale = False
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

		self.min_thresh_slider = QSlider(Qt.Horizontal, self)
		self.min_thresh_slider.setMinimum(0) # betweeen 100 and 200
		self.min_thresh_slider.setMaximum(100) # -for decimal places
		self.min_thresh_slider.setSingleStep(1)
		self.min_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height(), self.pixmap.width(), 20)
		
		self.max_thresh_slider = QSlider(Qt.Horizontal, self)
		self.max_thresh_slider.setMinimum(100)
		self.max_thresh_slider.setMaximum(200)
		self.max_thresh_slider.setSingleStep(1)
		self.max_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height()+self.min_thresh_slider.height(), self.pixmap.width(), 20)

	def updateSlider(self):
		self.slice = self.slider.value()
		self.update()

	def update_min_thresh(self):
		self.min_thresh = self.min_thresh_slider.value() # divide by 100 to get decimal points
		self.update()

	def update_max_thresh(self):
		self.max_thresh = self.max_thresh_slider.value() # divide by 100 to get decimal points
		self.update()


def mainViewer(stack):
	app = QApplication(sys.argv)
	win = mainView(stack)
	win.show()
	app.exec_()
	return

