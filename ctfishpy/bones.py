"""
CTreader is the main class you use to interact with ctfishpy
"""

from pathlib2 import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchio as tio
import warnings

class Bone(tio.Subject):
    def __init__(self) -> None:
        self.name = "BONE"
        self.n_classes = [None]
        self.class_names = []
        self.centers
        pass

    def localise(self):
        # TODO bring master.bone_centers here
        
        return

    def predict(self):
        pass

class Otolith(Bone):
    def __init__(self) -> None:
        super().__init__()
        self.name = "BONE"
        self.n_classes = [None]
        self.class_names = []
        self.centers

    def localise(self):
        return super().localise()

    def predict(self):
        # return super().predict()

        """
        TODO bring scripts/otoliths/pred_all to here

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


class Jaw(Bone):
    def __init__(self) -> None:
        super().__init__()
        self.name = "JAW"
        self.n_classes = [None]
        self.class_names = []
        self.centers

    def localise(self):
        return super().localise()

    def predict(self):
        """
        jaw predict 
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'predicting on {device}')

        jaw_nclasses = 5

        if model is None:
            start_filters = 32
            n_blocks = 5
            start = int(math.sqrt(start_filters))
            channels = [2**n for n in range(start, start + n_blocks)]
            strides = [2 for _ in range(1, n_blocks)]
            model = monai.networks.nets.AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=jaw_nclasses,
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

        grid_sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap, padding_mode='mean')
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
        label = undo_one_hot(label, jaw_nclasses, threshold)

        # TODO norm brightness histo?
        return label