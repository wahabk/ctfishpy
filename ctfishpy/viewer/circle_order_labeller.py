from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
from ..controller import *
from .mainviewer import mainView
import numpy as np
import cv2
import sys

class order_labeller(QApplication):
	def __init__(self, stack, circles):
		self.stack = stack
		self.circles = circles
		self.ordered_circles = []
		self.qlabel = QLabel(self)
		self.initUI()

	def update(self):

		self.image = self.stack

		# transform image to qimage and set pixmap
		self.image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.qlabel.setPixmap(self.pixmap)

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy: please click on the circles in order')
		self.update()
		self.qlabel.mousePressEvent = self.getPixel
		self.resize(self.pixmap.width(), self.pixmap.height())

	def getPixel(self , event):
		#get pixels of every click and assign circle order
		x = event.pos().x()
		y = event.pos().y()
		self.assign_circle_order(x, y)

	def assign_circle_order(self, x, y):
		#check if the click is in a circle 
		#add to ordered_circles and deleted from original circles
		for i in range(0, len(self.circles)):
			a, b, r = self.circles[i]
			if self.inside_circle(x, y, a, b, r):
				self.ordered_circles.append([a, b, r])
				self.circles = np.delete(self.circles, i, 0)
				break

	def inside_circle(self, x, y, a, b, r):
		#check if point x, y is in circle with centre and radius a, b ,r 
		return (x - a)*(x - a) + (y - b)*(y - b) < r*r
	
	def number_image(self, stack):
		num = 1
		for x, y, r in self.ordered_circles:
			for slice_ in stack:
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(slice_, str(num), (x,y), font, 4, (255,255,255), 2, cv2.LINE_AA)
			num = num+1
		return stack

	def get_order(self):
		return np.array(self.ordered_circles), self.number_image(self.og_stack)


def circle_order_labeller(labelled_img, circles):
	app = QApplication(sys.argv)
	ex = order_labeller(labelled_img, circles)
	ex.show()
	app.exec_()
	return ex.get_order()

if __name__ == "__main__":

	CTreader = CTreader()

	for i in range(0,1):
		ct, color = CTreader.read_dirty(i, r=(0,20))
		circle_dict  = CTreader.find_tubes(color)
		
	app = QApplication(sys.argv)
	ex = order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
	sys.exit(app.exec_())
