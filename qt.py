from qtpy.QtWidgets import QMessageBox, QApplication, QWidget, QPushButton, QToolTip, QLabel
from qtpy.QtGui import QFont, QPixmap, QImage
from qtpy.QtCore import Qt
from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

class window(QWidget):

	def __init__(self, img, circles):
		super().__init__()
		
		self.image = self.np2qt(img)
		self.circles = circles
		self.initUI()
		self.ordered_circles = []

	def initUI(self):
		self.setWindowTitle('CTFishPy')

		pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.label = QLabel(self)
		self.label.setPixmap(pixmap)
		self.label.mousePressEvent = self.getPixel

		self.resize(pixmap.width(), pixmap.height())
		self.show()

	def getPixel(self , event):
		x = event.pos().x()
		y = event.pos().y()
		self.assign_circle_order(x, y)

	def assign_circle_order(self, x, y):
		for a, b, r in self.circles:
			if self.inside_circle(x, y, a, b, r):
				self.ordered_circles.append([a, b, r])
			else:
				print('no circle in these coordinates')
		print(self.ordered_circles)

	def inside_circle(self, x, y, a, b, r):
		return (x - a)*(x - a) + (y - b)*(y - b) < r*r
	
	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

	def np2qt(self, img):
		height, width, channel = img.shape
		bytesPerLine = 3 * width
		return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
		
	def qt2np(self):
		pass

if __name__ == "__main__":

	CTreader = CTreader()

	for i in range(0,1):
		ct, color = CTreader.read_dirty(i, r=(0,20))
		circle_dict  = CTreader.find_tubes(color[10])

		#CTreader.view(ct) 
		print(circle_dict['circles'])
		if circle_dict['labelled_img'] is not None:
			pass
			cv2.imshow('output', circle_dict['labelled_img'])
			k = cv2.waitKey()

	app = QApplication(sys.argv)
	ex = window(circle_dict['labelled_img'], circle_dict['circles'])
	sys.exit(app.exec_())
	