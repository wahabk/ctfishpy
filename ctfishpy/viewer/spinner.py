from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
import numpy as np
import cv2
import sys

def rotate_image(image, angle, center=None):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	if center: image_center = center
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

class mainViewer(QMainWindow):

	def __init__(self, stack, center, label = None, thresh = False):
		super().__init__()
		self.thresh = thresh
		self.label = label
		self.center = center
		# convert 16 bit grayscale to 8 bit
		# by mapping the data range to 0 - 255
		if stack.dtype == 'uint16':
			self.stack = ((stack - stack.min()) / (stack.ptp() / 255.0)).astype(np.uint8) 
		else:
			self.stack = stack
		self.stack_length = self.stack.shape[0]

		self.initUI()


	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		self.spinner = Spinner(self.stack, center=self.center, label = self.label, parent = self, thresh = self.thresh)
		self.setCentralWidget(self.spinner)
		#widget.findChildren(QWidget)[0]

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, self.spinner.width(), self.spinner.height())
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

	def updateStatusBar(self, slice_ = 0):
		self.statusBar().showMessage(f'{slice_}/{self.stack_length}')


class Spinner(QWidget):

	def __init__(self, stack, center, label = None, thresh = False, stride = 1, parent = None):
		super().__init__()

		# init variables
		if np.max(stack) == 1: stack = stack*255 #fix labels
		self.og_image = stack
		self.max_angle = 365
		self.stride = stride
		self.angle = 0
		self.parent = parent
		self.min_thresh = 0
		self.max_thresh = 200
		self.thresh = thresh
		self.label = label
		self.pad = 20

		# label stack
		if self.label is not None:
			self.thresh == False
			# unpack images to convert each one to grayscale
			self.og_image = np.array([np.stack((img,)*3, axis=-1) for img in self.og_image])
			# change pixels to red if in label
			self.og_image[label == 1, :] = [255, 0, 0]

		#if self.og_image.shape[0] == self.og_image.shape[1]: self.is_single_image = True
		if len(self.og_image.shape) == 2: self.is_single_image = True
		else: self.is_single_image = False


		# set background colour to cyan
		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.qlabel = QLabel(self)
		self.qlabel.setMargin(10)
		self.initUI()

	def initUI(self):
		pad = self.pad
		self.update()
		if self.is_single_image == False:
			self.initSlider()
			self.slider.valueChanged.connect(self.updateSlider)
			self.setGeometry(0, 0, 
				self.pixmap.width() + pad, 
				self.pixmap.height() + pad + self.slider.height()*2+ pad*2)

		elif self.is_single_image: 
			self.initSlider()
			self.slider.valueChanged.connect(self.updateSlider)
			self.setGeometry(0, 0, 
				self.pixmap.width() + pad, 
				self.pixmap.height() + pad + self.slider.height()*2 + pad*2)

		if self.thresh == True: 
			self.initThresholdSliders()
			self.setGeometry(0, 0, 
				self.pixmap.width() + pad, 
				self.pixmap.height()+ pad + self.slider.height()*2 + self.min_thresh_slider.height()*2 + self.max_thresh_slider.height()*2)
			self.min_thresh_slider.valueChanged.connect(self.update_min_thresh)
			self.max_thresh_slider.valueChanged.connect(self.update_max_thresh)

		
			
	def update(self):

		if self.angle > self.max_angle-1: 	self.angle = 0
		if self.angle < 0: 					self.angle = self.max_angle-1
		
		self.image = rotate_image(self.og_image, self.angle)
		if self.thresh == True: 
			ret, self.image  = cv2.threshold(self.image, 
			self.min_thresh, self.max_thresh, cv2.THRESH_BINARY)

		# transform image to qimage and set pixmap
		self.image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.qlabel.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		#scroll through slices and go to beginning if reached max
		self.angle = self.angle + int(event.angleDelta().y()/120)*self.stride
		if self.angle > self.max_angle-1: 	self.angle = 0
		if self.angle < 0: 					self.angle = self.max_angle-1
		self.slider.setValue(self.angle)
		self.update()

	def np2qt(self, image):
		# transform np cv2 image to qt format

		# check length of image shape to check if image is grayscale or color
		if len(image.shape) == 2: grayscale = True
		elif len(image.shape) == 3: grayscale = False
		else: raise ValueError('[Viewer] Cant tell if stack is color or grayscale, weird shape :/')
		if self.is_single_image: grayscale = True


		# convert npimage to qimage depending on color mode
		if grayscale == True:
			height, width = self.image.shape
			bytesPerLine = width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		else:
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

	def initSlider(self):
		self.slider = QSlider(Qt.Horizontal, self)
		self.slider.setMinimum(0)
		self.slider.setMaximum(self.max_angle)
		self.slider.setGeometry(10, self.pixmap.height()+10, self.pixmap.width(), 20)

	def updateSlider(self):
		self.angle = self.slider.value()
		self.update()

	def initThresholdSliders(self):
		self.min_thresh_slider = QSlider(Qt.Horizontal, self)
		self.min_thresh_slider.setMinimum(0) # betweeen 100 and 200
		self.min_thresh_slider.setMaximum(150) # -for decimal places
		self.min_thresh_slider.setSingleStep(1)
		self.min_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height(), self.pixmap.width(), 20)
		
		self.max_thresh_slider = QSlider(Qt.Horizontal, self)
		self.max_thresh_slider.setMinimum(100)
		self.max_thresh_slider.setMaximum(255)
		self.max_thresh_slider.setSingleStep(1)
		self.max_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height()+self.min_thresh_slider.height(), self.pixmap.width(), 20)

	def update_min_thresh(self):
		self.min_thresh = self.min_thresh_slider.value() # divide by 100 to get decimal points
		self.update()

	def update_max_thresh(self):
		self.max_thresh = self.max_thresh_slider.value() # divide by 100 to get decimal points
		self.update()


def spinner(stack, center, label, thresh):
	app = QApplication(sys.argv)
	win = mainViewer(stack, center, label, thresh)
	win.show()
	app.exec_()
	return win.spinner.angle

'''
				listOfCoordinates = list(zip(indices[0], indices[1]))
				if listOfCoordinates != []:
					for (x, y) in listOfCoordinates:
						print(x, y)
									megastack = zip(self.og_image, self.label)
			labelled_stack = []
			for (img, l) in megastack:
				#indices = np.where(l == 1)
				img[l == 1, 0] = 255

				labelled_stack.append(img)
			self.og_image = np.array(labelled_stack)
'''