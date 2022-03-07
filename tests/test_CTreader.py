import ctfishpy
import pytest
import napari
import numpy as np

def test_viewer():
    ctreader = ctfishpy.CTreader()

    scan, metadata = ctreader.read(40)

    viewer = napari.Viewer()
    viewer.add_image(scan)
    napari.run()

    assert scan.shape is not None
