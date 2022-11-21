import ctfishpy
import napari
import numpy as np

from napari.layers import Image, Layer, Labels
from magicgui import magicgui
from scipy.ndimage import zoom

from copy import deepcopy
import cv2

import pandas as pd

from skimage.segmentation import flood
from skimage import io


@magicgui(
    auto_call=True,
    threshold={"widget_type": "Slider", "max": 255, "min": 0},
    new_value={"widget_type": "SpinBox", "max": 255, "min": 0},
    TwoD={"widget_type": "CheckBox"},
    # seek={"widget_type": "CheckBox"},
    undo={"widget_type": "PushButton"},
    layout="Horizontal",
)
def labeller(
    layer: Layer,
    label_layer: Labels,
    threshold: int = 125,
    new_value: int = 1,
    TwoD: bool = False,
    seek=False,
    undo=False,
) -> None:  # reset_center:bool=False
    if layer is not None:
        if label_layer is not None:
            assert isinstance(layer.data, np.ndarray)  # it will be!
            assert isinstance(label_layer.data, np.ndarray)  # it will be!

            label = deepcopy(label_layer.data)
            image = layer.data

            point = layer.metadata["point"]
            _slice = layer.metadata["slice"]
            if point is not None:
                if len(point) == 3:
                    if TwoD == False:
                        point = tuple([int(x) for x in point])
                        new_label = None
                        new_label = flood(image, point, tolerance=threshold)

                        print(label.min(), label.max(), label.shape)
                        print(new_label.min(), new_label.max(), new_label.shape)

                        label_layer.data[new_label == True] = new_value
                        layer.metadata["history"] = np.concatenate(
                            [
                                layer.metadata["history"],
                                np.expand_dims(label_layer.data, 0),
                            ],
                            axis=0,
                        )
                        if len(layer.metadata["history"]) > 4:
                            layer.metadata["history"] = layer.metadata["history"][1:]

                    else:
                        dims_order = layer._dims_order
                        pos = layer.position
                        slice_ = int(point[dims_order[0]])  # pos[dims_order[0]]

                        point = tuple([int(x) for x in point])
                        point = tuple([point[d] for d in dims_order[1:]])
                        # label = label[_slice]
                        image = get_from_index(dims_order[0], image, slice_)
                        label = get_from_index(dims_order[0], label, slice_)
                        # np.squeeze(np.take(label, dims_order, slice_))
                        new_label = None
                        new_label = flood(image, point, tolerance=threshold)

                        # print(layer.data.shape, dims_order, pos, slice_, point)
                        # print(image.shape, label.shape)
                        # print(label.min(), label.max(), label.shape)
                        # print(new_label.min(), new_label.max(), new_label.shape)

                        zeros = np.zeros_like(label_layer.data)
                        zeros[zeros == 0] = False
                        zeros = put_in_index(dims_order[0], zeros, slice_, new_label)

                        label_layer.data[zeros == True] = new_value
                        layer.metadata["history"] = np.concatenate(
                            [
                                layer.metadata["history"],
                                np.expand_dims(label_layer.data, 0),
                            ],
                            axis=0,
                        )
                        if len(layer.metadata["history"]) > 3:
                            layer.metadata["history"] = layer.metadata["history"][1:]

    if seek == False:
        layer.metadata["point"] = None
    return


def get_from_index(order: int, arr: np.ndarray, index: int):
    if order == 0:
        return arr[index, :, :]
    if order == 1:
        return arr[:, index, :]
    if order == 2:
        return arr[:, :, index]


def put_in_index(order: int, arr: np.ndarray, index: int, b: np.ndarray):
    if order == 0:
        arr[index, :, :] = b
        return arr
    if order == 1:
        arr[:, index, :] = b
        return arr
    if order == 2:
        arr[:, :, index] = b
        return arr


def create_labeller(viewer, layer, label_layer) -> None:
    widget = labeller
    # layer.metadata['point'] = None

    viewer.window.add_dock_widget(widget, name="labeller", area="right")
    viewer.layers.events.changed.connect(widget.reset_choices)

    # TODO current slice?
    # TODO add only in

    @layer.mouse_drag_callbacks.append
    def get_event(layer, event):
        if event.button == 2:  # if left click
            layer.metadata["point"] = event.position  # flip because qt :(
            # layer.metadata['slice'] = int(layer.position[0]) # get the slice youre looking at
            widget.update()
        return

    @widget.undo.clicked.connect
    def undo():
        print("CHECKING")
        if len(layer.metadata["history"]) > 1:
            print("UNDOING")
            print(layer.metadata["history"].shape)
            print(layer.metadata["history"][-1].shape)
            print(label_layer.data.shape)
            label_layer.data = layer.metadata["history"][-2]
            layer.metadata["history"] = layer.metadata["history"][:-1]
            widget.update()
        return

    return


def label(scan):
    l = np.zeros_like(scan)

    viewer = napari.Viewer()
    layer = viewer.add_image(scan)
    layer.metadata = {"point": None, "history": np.stack([l]), "slice": 0}
    label_layer = viewer.add_labels(l)

    create_labeller(viewer, layer, label_layer)

    viewer.show(block=True)

    return label_layer.data


def insert_a_in_b(a: np.ndarray, b: np.ndarray, center=None):

    roiZ, roiY, roiX = a.shape
    zl, yl, xl = int(roiZ / 2), int(roiY / 2), int(roiX / 2)

    if center is None:
        b_center = [int(b.shape[2] / 2), int(b.shape[3] / 2), int(b.shape[4] / 2)]
        z, y, x = b_center
    else:
        z, y, x = center
    z, y, x = int(z), int(y), int(x)

    print(z, y, x, center, zl, yl, xl)
    print(a.shape, b.shape)
    print(z - zl, z + zl, y - yl, y + yl, x - xl, x + xl)

    b[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl] = a

    return b


if __name__ == "__main__":
    # dataset_path = "/home/ak18001/Data/HDD/uCT"
    dataset_path = "/home/wahab/Data/HDD/uCT"

    ctreader = ctfishpy.CTreader(dataset_path)

    bone = "JAW_manual"
    roiSize = (256, 256, 320)

    sample = pd.read_csv("output/results/jaw/training_sample_curated.csv")

    n = 257

    scan = ctreader.read(n)
    # scan = ctreader.to8bit(scan)
    center = ctreader.jaw_centers[n]
    scan = ctreader.crop3d(scan, roiSize=roiSize, center=center)
    # scan = scan[1000:]
    # print(scan.shape)
    # ctreader.view(scan)

    # label = label(scan)

    # print(lab.min(), lab.max(), lab.shape)
    label = io.imread(dataset_path + "/LABELS/JAW/jaw_257.tif")
    # TODO make ctreader write tif read tif
    # print(scan.shape, label.shape)

    # zeros =  np.zeros_like(scan)
    # zeros  = insert_a_in_b(label, zeros, center=center)
    # label=zeros
    # print(scan.shape, label.shape)

    ctreader.view(scan, label=label)
    # ctreader.write_label(bone, label, n)

    # TODO rewrite this one first as qt widget
    # TODO write only in class
    # TODO write sweep thresh
