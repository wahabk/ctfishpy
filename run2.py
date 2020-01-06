from tools.utility import IndexTracker
from tools.CTreader import CTreader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

CTreader = CTreader()
ct208 = CTreader.read(208)


