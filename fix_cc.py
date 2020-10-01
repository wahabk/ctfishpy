import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


if __name__ == "__main__":
	ctreader = ctfishpy.CTreader()
	projections = ctreader.get_max_projections(40)
	positions = ctfishpy.GUI.mainFixer(projections)
	print(positions)