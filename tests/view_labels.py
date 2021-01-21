import ctfishpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

ctreader = ctfishpy.CTreader()
n = 40
ct, stack_metadata = ctreader.read(n, align=True)
label = ctreader.read_label('Otoliths', n=n, align=True)


ctreader.view(ct, label=label)
