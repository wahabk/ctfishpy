from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
from ..controller import *
from .mainviewer import mainView
import numpy as np
import cv2
import sys

class MainWindow(QMainWindow):
	def __init__(self, labelled_img, circles):
		super().__init__()
		self.viewer = Viewer(labelled_img, circles)
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
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

class Viewer(QWidget):
	def __init__(self, image, circles):
		super().__init__()
		self.image = image
		self.circles = circles
		self.ordered_circles = []
		self.initUI()


	def initUI(self):
		#initialise UI
		self.label = QLabel(self)
		self.setWindowTitle('CTFishPy: please click on the circles in order')
		self.label.mousePressEvent = self.getPixel
		self.update()
		self.resize(self.pixmap.width(), self.pixmap.height())


	def update(self):
		# transform image to qimage and set pixmap
		self.pixmap_image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.pixmap_image))
		self.label.setPixmap(self.pixmap)

	def np2qt(self, image):
		# transform np cv2 image to qt format
		height, width, channel = image.shape
		bytesPerLine = 3 * width
		return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

	def getPixel(self , event):
		#get pixels of every click and assign circle order
		x = event.pos().x()
		y = event.pos().y()
		self.assign_circle_order(x, y)

	def assign_circle_order(self, x, y):
		#check if the click is in a circle 
		#add to ordered_circles and deleted from original circles
		for (a, b, r) in self.circles:
			if self.inside_circle(x, y, a, b, r):
				self.ordered_circles.append([a, b, r])

				self.image = self.number_image(self.image, len(self.ordered_circles), (a,b))
				break
		self.update()

	def inside_circle(self, x, y, a, b, r):
		#check if point x, y is in circle with centre and radius a, b ,r 
		return (x - a)*(x - a) + (y - b)*(y - b) < r*r

	def number_image(self, img, num, loc):

		font = cv2.FONT_HERSHEY_SIMPLEX
		return cv2.putText(img=img.copy(), text=str(num), org=loc, fontFace=font, fontScale=4, color=(255,255,255), thickness=2)


def circle_order_labeller(labelled_img, circles):
	app = QApplication(sys.argv)
	win = MainWindow(labelled_img, circles)
	win.show()
	app.exec_()
	return np.array(win.viewer.ordered_circles), win.viewer.image
