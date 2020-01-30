from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import numpy as np
import cv2
from qtpy.QtWidgets import QMessageBox, QApplication, QWidget, QPushButton, QToolTip
from qtpy.QtGui import QIcon, QFont

import sys

def qthello():
	app = QApplication(sys.argv)
	button = QPushButton("Hello World", None)
	button.show()
	app.exec_()

def emptywindow():
	app = QApplication(sys.argv)

	w = QWidget()
	w.resize(500, 500)
	w.move(500, 500)
	w.setWindowTitle('CTFishPy')
	w.show()

	sys.exit(app.exec_())



class iconnedwindow(QWidget):
	
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setToolTip('This is a <b>QWidget</b> widget')

		btn = QPushButton('Button', self)
		btn.setToolTip('This is a <b>QPushButton</b> widget')
		btn.resize(btn.sizeHint())
		btn.move(50, 50)

		self.setGeometry(700, 700, 500, 500)
		self.setWindowTitle('Iconned')
		self.setWindowIcon(QIcon('~/Pictures/zebrafish.png'))
		
		self.show()

	def closeEvent(self, event):
		reply = QMessageBox.question(self, 'Message',
			"Are you sure to quit?", QMessageBox.Yes | 
			QMessageBox.No, QMessageBox.Yes)

		if reply == QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()  

CTreader = CTreader()
master = CTreader.mastersheet()
for i in range(0,1):
	ct, color = CTreader.read_dirty(i, r=(1000,1010))
	circle_dict  = CTreader.find_tubes(color[0])

	#CTreader.view(ct) 

	if circle_dict['labelled_img'] is not None:
		pass
		#cv2.imshow('output', circle_dict['labelled_img'])
		#cv2.waitKey()

app = QApplication(sys.argv)
ex = iconnedwindow()
sys.exit(app.exec_())