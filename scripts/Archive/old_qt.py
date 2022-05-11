from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import numpy as np
import cv2
from qtpy.QtWidgets import QMessageBox, QApplication, QWidget, QPushButton, QToolTip, QLabel
from qtpy.QtGui import QFont, QPixmap, QImage
import sys

class window(QWidget):

	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		#self.setGeometry(700, 700, 500, 500)

		self.image = QImage('zebrafish.png')
		pixmap = QPixmap(QPixmap.fromImage(self.image))

		self.label = QLabel(self)
		self.label.setPixmap(pixmap)
		self.label.mousePressEvent = self.getPixel

		self.setWindowTitle('CTFishPy')
		self.resize(pixmap.width(), pixmap.height())

		self.show()

	def getPixel(self , event):
	    x = event.pos().x()
	    y = event.pos().y()
	    print(x,y)

app = QApplication(sys.argv)
ex = window()
sys.exit(app.exec_())


'''
	def closeEvent(self, event):
		reply = QMessageBox.question(self, 'Message',
			"Are you sure to quit?", QMessageBox.Yes | 
			QMessageBox.No, QMessageBox.Yes)

		if reply == QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()

		#btn = QPushButton('Button', self)
		#btn.setToolTip('This is a <b>QPushButton</b> widget')
		#btn.resize(btn.sizeHint())
		#btn.move(50, 200)
'''

