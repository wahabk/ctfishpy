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
        self.name = "BONE"
        pass

    def localise(self):
        pass

    def predict(self):
        pass


class Otolith(Bone):
    def __init__(self) -> None:
        super().__init__()

    def localise(self):
        return super().localise()

    def predict(self):
        # return super().predict()

        """
        helper function for testing
        """

        bone = "OTOLITH"
        n_blocks = 3
        start_filters = 32
        roi = (128,128,160)
        n_classes = 4
        batch_size = 6
        num_workers = 10

        start = int(math.sqrt(start_filters))
        channels = [2**n for n in range(start, start + n_blocks)]
        strides = [2 for n in range(1, n_blocks)]

        # model
        if model is None:
            model = model = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=4,
                channels=channels,
                strides=strides,
                num_res_units=n_blocks,
                act="PRELU",
                norm="INSTANCE",
            )

        model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

        if weights_path is not None:
            model_weights = torch.load(weights_path, map_location=device) # read trained weights
            model.load_state_dict(model_weights) # add weights to model

        pred_loader = CTDatasetPredict(dataset_path, bone=bone, indices=nums, roi_size=roi, n_classes=n_classes)
        pred_loader = torch.utils.data.DataLoader(pred_loader, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'predicting on {device}')

        predict_list = []
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(pred_loader):
                x = batch
                x = x.to(device)

                out = model(x)  # send through model/network
                out = torch.softmax(out, 1)

                # post process to numpy array
                # array = x.cpu().numpy()[0,0] # 0 batch, 0 class
                y_pred = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray

                print(f"appending this pred index {idx} shape {y_pred.shape}")

                predict_list.append(y_pred)

        return predict_list