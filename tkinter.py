from CTFishPy.CTreader import CTreader
import CTFishPy.utility as utility
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5 import QtCore
import sys
pd.set_option('display.max_rows', None)

CTreader = CTreader()
master = CTreader.mastersheet()

for i in range(0,1):
	ct, color = CTreader.read_dirty(i, r=(1000,1020))
	circle_dict  = CTreader.find_tubes(color[10])

	#CTreader.view(ct) 

	if circle_dict['labelled_img'] is not None:
		pass
		#cv2.imshow('output', circle_dict['labelled_img'])
		#cv2.waitKey()


cropped_cts = CTreader.crop(color, circle_dict['circles'])






def getPixel(self, event):
    x = event.pos().x()
    y = event.pos().y()
    c = self.img.pixel(x,y)  # color code (integer): 3235912

    return x, y


self.img = QtGui.QImage(circle_dict['labelled_img'])
pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(self.img))
img_label = QtGui.QLabel()
img_label.QtGui.setPixmap(pixmap)
img_label.mousePressEvent = self.getPixel


def qthello():
	app = QtWidgets.QApplication(sys.argv)
	button = QtWidgets.QPushButton("Hello World", None)
	button.show()
	app.exec_()

qthello()


#Rename folders using python os.rename

#cv2.imshow('output', cropped_cts[0][0])
#cv2.waitKey()

