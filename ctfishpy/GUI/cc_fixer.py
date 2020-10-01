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

		# convert 16 bit grayscale to 8 bit
		# by mapping the data range to 0 - 255
		self.stack = stack
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
		self.setGeometry(1100, 10, self.fixer.width(), self.fixer.height())

		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

	def updateStatusBar(self, slice_ = 0):
		self.statusBar().showMessage(f'{slice_}/{self.stack_length}')


class Fixer(QWidget):

	def __init__(self, projections, parent):
		super().__init__()

		# init variables
		for img in projections:
			if np.max(img) == 1: 
				img = img*255 #fix labels so you can view them if are main stack
			print(img.shape)
		self.ogstack = projections
		self.parent = parent
		self.pad = 20
		self.step = 0
		self.position_list = []
		'''
		position_list = [
			[x, y],
			[z, x],
			[z, y]
		]
		'''

		#if self.ogstack.shape[0] == self.ogstack.shape[1]: self.is_single_image = True
		self.is_single_image = True

		# set background colour to cyan
		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.qlabel = QLabel(self)
		self.qlabel.setMargin(10)
		self.initUI()

	def initUI(self):
		self.update()
		
		self.qlabel.mousePressEvent = self.getPixel

	def update(self):

		self.image = self.ogstack[self.step]
		self.setGeometry(0, 0, 
			self.image.shape[0] + self.pad, 
			self.image.shape[1] + self.pad*3)

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

		self.position_list.append([x, y])
		self.step += 1
		if self.step == 3:
			self.finish()
			return
		self.update()
			

	def finish(self):
		self.final_position = self.position_list
		self.parent.close()


def mainFixer(stack):
	app = QApplication(sys.argv)
	win = mainView(stack)
	win.show()
	app.exec_()
	return win.fixer.final_position
