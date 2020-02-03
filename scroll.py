from qtpy.QtWidgets import QMessageBox, QApplication, QWidget, QPushButton, QToolTip, QLabel
from qtpy.QtGui import QFont, QPixmap, QImage
from qtpy.QtCore import Qt, QTimer
from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

class window(QWidget):

	def __init__(self, img, circles):
		super().__init__()
		self.image = img
		self.circles = circles
		self.ordered_circles = []
		self.slice = 0
		self.label = QLabel(self)
		self.initUI()

	def initUI(self):
		self.setWindowTitle('CTFishPy')
		self.update()
		self.label.mousePressEvent = self.getPixel
		self.resize(self.pixmap.width(), self.pixmap.height())

	def update(self):
		self.image = self.np2qt(self.image[self.slice])
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.label.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		self.slice = self.slice + int(event.angleDelta().y()/120)
		self.update()

	def getPixel(self , event):
		x = event.pos().x()
		y = event.pos().y()
		self.assign_circle_order(x, y)

	def assign_circle_order(self, x, y):
		for i in range(0, len(self.circles)):
			a, b, r = self.circles[i]
			if self.inside_circle(x, y, a, b, r):
				self.ordered_circles.append([a, b, r])
				self.circles = np.delete(self.circles, i, 0)
				break

	def inside_circle(self, x, y, a, b, r):
		return (x - a)*(x - a) + (y - b)*(y - b) < r*r
	
	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

	def np2qt(self, img):
		height, width, channel = img.shape
		bytesPerLine = 3 * width
		return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
	
	def get_order(self):
		return self.ordered_circles

def labeller(labelled_img, circles):
	app = QApplication(sys.argv)
	ex = window(labelled_img, circles)
	ex.show()
	app.exec_()
	return ex.get_order()

if __name__ == "__main__":

	CTreader = CTreader()

	for i in range(0,1):
		ct, color = CTreader.read_dirty(i, r=(1000,1200))
		circle_dict  = CTreader.find_tubes(color[10])

	app = QApplication(sys.argv)
	ex = window(color, circle_dict['circles'])
	ex.show()
	app.exec_()
	