from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel
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
		self.slice = 0
		#self.label = QLabel(self)
		self.stack_size = stack.shape[0]-1

		#check length of image shape to check if image is grayscale or color
		if len(stack.shape) == 3: self.grayscale = True
		elif len(stack.shape) == 4: self.grayscale = False
		else: raise ValueError('[viewer] Cant tell if stack is color or grey scale')

		self.initUI()

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')

		self.resize(500, 700)
		self.update()
 
	def update(self):
		#Update displayed image
		self.image = self.np2qt(self.npstack[self.slice])
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.label.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		#scroll through slices and go to beginning if reached max
		self.slice = self.slice + int(event.angleDelta().y()/120)*10
		if self.slice > self.stack_size: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size
		#self.update()
	
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

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
	

def mainwin(stack):
	app = QApplication(sys.argv)
	win = MainWindow(stack)
	win.show()
	app.exec_()


if __name__ == "__main__":
	CTreader = CTreader()

	ct, color = CTreader.read_dirty(1, r=(0,20))
	circle_dict  = CTreader.find_tubes(color)
	
	mainwin(ct)