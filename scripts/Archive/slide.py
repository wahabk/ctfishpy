from qtpy.QtWidgets import QApplication
from .. import CTreader
from . import view
import numpy as np
import cv2
import sys

class slide(view.Window):
	def __init__(self):
		super().__init__(stack) #inherit methods from vie.wwindow
		self.og_stack = stack
		self.thresh = [0, 100]
		self.initUI()

	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy: please click on the circles in order')
		self.update()
		self.resize(self.pixmap.width(), self.pixmap.height())

	def thresh(self , event):
		#get pixels of every click and assign circle order
		x = event.pos().x()
		y = event.pos().y()
		self.assign_circle_order(x, y)

def slider(stack):
	app = QApplication(sys.argv)
	ex = slide(stack)
	ex.show()
	app.exec_()
	return

if __name__ == "__main__":

	CTreader = CTreader()

	for i in range(0,1):
		ct, color = CTreader.read_dirty(i, r=(0,20))
		circle_dict  = CTreader.find_tubes(color)
		
	app = QApplication(sys.argv)
	ex = order_labeller(circle_dict['labelled_stack'], circle_dict['circles'])
	sys.exit(app.exec_())
