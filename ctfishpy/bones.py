"""
CTreader is the main class you use to interact with ctfishpy
"""

from copy import deepcopy
from sklearn.utils import deprecated
from .read_amira import read_amira
from pathlib2 import Path
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import h5py
import json
import napari
import warnings
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import datetime
import os
import time


class Bone():
    def __init__(self) -> None:
        pass

    def localise(self):
        pass

    def predict(self):
        pass