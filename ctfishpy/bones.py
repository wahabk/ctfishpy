"""
CTreader is the main class you use to interact with ctfishpy
"""

import torch
from CTreader import CTreader
from train_utils import CTDatasetPredict, undo_one_hot
from tqdm import tqdm
import numpy as np
import torchio as tio
import math
import monai
import warnings
from pathlib2 import Path

class Bone(tio.Subject):
    def __init__(self) -> None:
        self.name = "BONE"
        self.n_classes = None #including background
        self.class_names = []
        self.roi_size = ()
        self.centers = []
        pass

    def localise(self):
        # TODO bring master.bone_centers here
        return

    def predict(self, array, weights_path=None, model=None,):
        pass

class Otolith(Bone):
    def __init__(self) -> None:
        super().__init__()
        self.name = "OTOLITHS"
        self.n_classes = 4 
        self.class_names = ["Lagenar", "Saccular", "Utricular"]
        self.roi_size = (128,128,160)
        self.centers = []

    def localise(self):
        return super().localise()

    def predict(self, array, weights_path=None, model=None,):
        # return super().predict()

        """
        NOTE array size must be 128x128x160

        TODO bring scripts/otoliths/pred_all to here

        helper function for testing
        """

        n_blocks = 3
        start_filters = 32

        start = int(math.sqrt(start_filters))
        channels = [2**n for n in range(start, start + n_blocks)]
        strides = [2 for n in range(1, n_blocks)]

        # model
        if model is None:
            model = model = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.n_classes,
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

        X = array
        X = np.array(X/X.max(), dtype=np.float32)
        X = np.expand_dims(X, 0)      # if numpy array
        X = torch.from_numpy(X)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'predicting on {device}')

        predict_list = []
        model.eval()
        with torch.no_grad():
            X = X.to(device)

            out = model(X)  # send through model/network
            out = torch.softmax(out, 1)

            # post process to numpy array
            # array = x.cpu().numpy()[0,0] # 0 batch, 0 class
            y_pred = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray

        label = undo_one_hot(y_pred, self.n_classes)

        return label


class Jaw(Bone):
    def __init__(self) -> None:
        super().__init__()
        self.name = "JAW"
        self.n_classes = 5
        self.class_names = ["L_Dentary", "R_Dentary", "L_Quadrate", "R_Quadrate"]
        self.roi_size = (160,160,160)
        self.patch_overlap=(16,16,16)
        self.centers = []

    def localise(self):
        return super().localise()

    def predict(self, array, weights_path=None, model=None,):
        """
        jaw predict 
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'predicting on {device}')

        if model is None:
            start_filters = 32
            n_blocks = 5
            start = int(math.sqrt(start_filters))
            channels = [2**n for n in range(start, start + n_blocks)]
            strides = [2 for _ in range(1, n_blocks)]
            model = monai.networks.nets.AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.n_classes,
                channels=channels,
                strides=strides,
            )

        model = torch.nn.DataParallel(model, device_ids=None)
        model.to(device)

        if weights_path is not None:
            model_weights = torch.load(weights_path, map_location=device) # read trained weights
            model.load_state_dict(model_weights) # add weights to model

        # The weights require dataparallel because it's used in training
        # But dataparallel doesn't work on cpu so remove it if need be
        if device == "cpu": model = model.module.to(device)
        elif device == "cuda": model = model.to(device)

        array = np.array(array/array.max(), dtype=np.float32)
        array = np.expand_dims(array, 0)
        X = torch.from_numpy(array)

        subject = tio.Subject(
            ct=tio.ScalarImage(tensor=X),
        )

        grid_sampler = tio.inference.GridSampler(subject, patch_size=self.roi_size, patch_overlap=self.patch_overlap, padding_mode='mean')
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        with torch.no_grad():
            for i, patch_batch in tqdm(enumerate(patch_loader)):
                input_ = patch_batch['ct'][tio.DATA]
                locations = patch_batch[tio.LOCATION]

                input_= input_.to(device)

                out = model(input_)  # send through model/network
                out = torch.softmax(out, 1)

                aggregator.add_batch(out, locations)

        output_tensor = aggregator.get_output_tensor()

        # post process to numpy array
        label = output_tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
        label = undo_one_hot(label, self.n_classes)

        return label