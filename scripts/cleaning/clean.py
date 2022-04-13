import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib2 import Path
import napari


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	lump = ctfishpy.Lumpfish()
	master = ctreader.mastersheet()

	scan, metadata = ctreader.read(468, align=False)
	ctreader.view(scan)
	scan, metadata = ctreader.read(468, align=True)
	ctreader.view(scan)


	viewer = napari.Viewer(show=False)
	angle, center = lump.spin(viewer, scan)
	print(angle, center)

