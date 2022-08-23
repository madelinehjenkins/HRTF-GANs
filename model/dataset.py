import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/7292452634137d8f5d4478e44727ec1166a89125/dataset.py
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates
from preprocessing.convert_coordinates import convert_sphere_to_cube
from preprocessing.utils import calc_hrtf


class TrainValidHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        hrtf_dir (str): Train/Valid dataset address.
        upscale_factor (int): hrtf up scale factor.
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, hrtf_dir: str, upscale_factor: int, transform=None, validation=False) -> None:
        super(TrainValidHRTFDataset, self).__init__()
        # Get all hrtf file names in folder
        self.hrtf_file_names = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in os.listdir(hrtf_dir)]
        # How many times the high-resolution hrtf is the low-resolution hrtf
        self.upscale_factor = upscale_factor
        # transform to be applied to the data
        self.transform = transform
        # coordinates for sphere
        self.validation = validation
        if validation:
            sphere_path = os.path.join(hrtf_dir, "..", "sphere_coords.pickle")
            with open(sphere_path, "rb") as file:
                self.hr_sphere = pickle.load(file)
            self.lr_sphere = torch.permute(
                torch.nn.functional.interpolate(
                    torch.permute(self.hr_sphere, (3, 0, 1, 2)),
                    scale_factor=1 / self.upscale_factor), (1, 2, 3, 0))
            self.hr_list = [item for panel in self.hr_sphere.tolist() for x in panel for item in x]
            self.lr_list = [item for panel in self.lr_sphere.tolist() for x in panel for item in x]
            self.barycentric_triangles, self.barycentric_coeffs = self.get_barycentric_coordinates()

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf, hrir = pickle.load(file)

        # hrtf processing operations
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            hr_hrir = torch.permute(hrir, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
            hr_hrir = torch.permute(self.transform(hr_hrir), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, panels, X, Y)
            hr_hrtf = torch.permute(hrtf, (3, 0, 1, 2))
            hr_hrir = torch.permute(hrir, (3, 0, 1, 2))

        # downsample hrtf
        lr_hrtf = torch.nn.functional.interpolate(hr_hrtf, scale_factor=1 / self.upscale_factor)
        lr_hrir = torch.nn.functional.interpolate(hr_hrir, scale_factor=1 / self.upscale_factor)

        if self.validation:
            interpolated_hrir = []
            for i, p in enumerate(self.hr_list):
                coeffs = self.barycentric_coeffs[i]
                features = self.get_triangle_features(self.barycentric_triangles[i], lr_hrir)

                hrir_p = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]
                interpolated_hrir.append(hrir_p)
            interpolated_magnitudes, _ = calc_hrtf(interpolated_hrir)
            # create empty list of lists of lists and initialize counter
            edge_len = 16
            magnitudes_raw = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
            count = 0

            for elevation, azimuth in self.hr_list:
                panel, x, y = convert_sphere_to_cube(elevation, azimuth)
                # based on cube coordinates, get indices for magnitudes list of lists
                PI_4 = np.pi / 4
                i = panel - 1
                j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
                k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))

                # add to list of lists of lists and increment counter
                magnitudes_raw[i][j][k] = interpolated_magnitudes[count]

                count += 1
            hr_hrtf_barycentric = torch.permute(torch.tensor(np.array(magnitudes_raw)), (3, 0, 1, 2))

        else:
            hr_hrtf_barycentric = None

        return {"lr": lr_hrtf, "hr": hr_hrtf, "hr_barycentric": hr_hrtf_barycentric,
                "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)

    def get_barycentric_coordinates(self):
        barycentric_triangles = []
        barycentric_coeffs = []

        for i in range(5*16*16):
            hr_loc = self.hr_list[i]
            triangle_vertices = get_triangle_vertices(elevation=hr_loc[0], azimuth=hr_loc[1],
                                                      sphere_coords=self.lr_list)
            coeffs = calc_barycentric_coordinates(elevation=hr_loc[0], azimuth=hr_loc[1],
                                                  closest_points=triangle_vertices)
            barycentric_triangles.append(triangle_vertices)
            barycentric_coeffs.append(coeffs)

        return barycentric_triangles, barycentric_coeffs

    def get_triangle_features(self, triangle_vertices, lr_hrir):
        lr_hrir_list = [item for panel in torch.permute(lr_hrir, (1, 2, 3, 0)).tolist() for x in panel for item in x]
        features = []
        for p in triangle_vertices:
            index = self.lr_list.index(list(p))
            feature_p = lr_hrir_list[index]
            features.append(np.array(feature_p))
        return features


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
