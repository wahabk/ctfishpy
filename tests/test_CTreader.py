import ctfishpy
import pytest
import napari
import numpy as np

def test_viewer():
    ctreader = ctfishpy.CTreader()

    scan = ctreader.read(40)

    viewer = napari.Viewer()
    viewer.add_image(scan)
    napari.run()

    assert isinstance(scan, np.ndarray)
