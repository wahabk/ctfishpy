import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import napari
import cv2


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()
	master = ctreader.mastersheet()

	# scan, metadata = ctreader.read(468, align=True)
	# ctreader.view(scan)


	# viewer = napari.Viewer(show=False)
	# angle, center = lump.spin(viewer, scan)
	# print(angle, center)

	# TODO write dicom

	array = ctreader.read_dicom('examples/Data/image1.dcm')
	print(array.shape)
	array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	ctreader.view(array)

