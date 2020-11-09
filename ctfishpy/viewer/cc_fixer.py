from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout, QScrollArea
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor, QPalette
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
import numpy as np
import cv2
import sys

class mainView(QMainWindow):

	def __init__(self, stack):
		super().__init__()

		self.stack = stack
		self.pad = 20
		self.initUI()


	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		self.fixer = Fixer(self.stack, self)
		self.scrollArea = QScrollArea()
		self.scrollArea.setWidget(self.fixer)
		self.scrollArea.setBackgroundRole(QPalette.Dark)
		self.layout = QVBoxLayout(self.scrollArea)
		self.setCentralWidget(self.scrollArea)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, self.fixer.width()+self.pad, self.fixer.height()+self.pad)

		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()


class Fixer(QWidget):

	def __init__(self, projection, parent):
		super().__init__()

		# init variables
		if np.max(projection) == 1: 
			projection = projection*255 #fix labels so you can view them if are main stack
		self.parent = parent
		self.pad = 20
		self.image = projection
		self.position = []


		# set background colour to cyan
		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.qlabel = QLabel(self)
		self.qlabel.setMargin(0)
		self.initUI()

	def initUI(self):
		
		self.setGeometry(10, 10, self.image.shape[1], self.image.shape[0])
		self.update()
		self.qlabel.mousePressEvent = self.getPixel

	def update(self):

		self.qlabel.setGeometry(10, 10, self.image.shape[1], self.image.shape[0])

		# transform image to qimage and set pixmap
		self.image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.qlabel.setPixmap(self.pixmap)
		

	def np2qt(self, image):
		# transform np cv2 image to qt format

		# check length of image shape to check if image is grayscale or color
		if len(image.shape) == 2: grayscale = True
		elif len(image.shape) == 3: grayscale = False
		else: raise ValueError('[Viewer] Cant tell if stack is color or grayscale, weird shape :/')

		# convert npimage to qimage depending on color mode
		if grayscale == True:
			height, width = self.image.shape
			bytesPerLine = width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		else:
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
	
	def getPixel(self , event):
		#get pixels of every click and assign circle order
		x = event.pos().x()
		y = event.pos().y()
		self.position = [x, y]
		self.parent.close()


def mainFixer(stack):
	app = QApplication(sys.argv)
	win = mainView(stack)
	win.show()
	app.exec_()
	return win.fixer.position
