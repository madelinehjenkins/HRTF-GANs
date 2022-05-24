import itertools
import pickle

import pandas as pd
import torch
from hrtfdata.torch.full import ARI, CHEDAR
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from matplotlib import cm
from collections import Counter

from barycentric_calcs import calc_all_distances, calc_barycentric_coordinates
from cubed_sphere import CubedSphere
from plot import plot_3d_shape, plot_flat_cube, plot_impulse_response, plot_interpolated_features
from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cube, convert_sphere_to_cartesian, \
    convert_cube_to_cartesian
from utils import get_feature_for_point, generate_euclidean_cube, triangle_encloses_point, get_possible_triangles, \
    calc_all_interpolated_features, save_euclidean_cube
from KalmanFilter import KalmanFilter as kf

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="last")  # temporary measure to avoid loading entire dataset each time


def normalize_smooth_trim_fade(hrir, threshold, pre_window, length):
    # normalize such that max(abs(hrir)) == 1
    plot_impulse_response(hrir, title="Original HRIR")
    rescaling_factor = 1 / max(np.abs(hrir))
    normalized_hrir = rescaling_factor * hrir
    # smooth HRIR with moving average
    smoothed_hrir = np.abs((normalized_hrir + [0]) + ([0] + normalized_hrir)) / 2
    smoothed_hrir[0] = 0
    plot_impulse_response(smoothed_hrir, title="Normalized and smoothed HRIR")
    # find first time HRIR exceeds some threshold
    over_threshold_index = list(smoothed_hrir).index(next(i for i in smoothed_hrir if i > threshold))
    # trim HRIR based on first time threshold is exceeded
    start = over_threshold_index - pre_window
    stop = start + length
    trimmed_hrir = smoothed_hrir[start:stop]
    plot_impulse_response(trimmed_hrir, title="Normalized, smoothed, and trimmed HRIR")
    # create fade window in order to taper off HRIR towards the end
    fadeout = np.arange(0.9, -0.1, -0.1).tolist()
    fade_window = [1] * (length - 10) + fadeout
    return trimmed_hrir * fade_window


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    print(ds[0]['features'].mask.shape)
    print(type(ds[0]['features'][1][0][0]))

    i = 70
    j = 10
    if not ds[0]['features'].mask[i][j][0]:
        hrir = ds[0]['features'][i][j]
        transformed_hrir = normalize_smooth_trim_fade(hrir, threshold=0.8, pre_window=4, length=30)
        plot_impulse_response(transformed_hrir, title="Normalized, smoothed, trimmed, and faded HRIR")
    # generate_euclidean_cube(cs.get_sphere_coords(), "euclidean_data_ARI", edge_len=2)
    #
    # with open("euclidean_data_ARI", "rb") as file:
    #     load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)
    #
    # plot_interpolated_features(cs, ds[0]['features'], 30, load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs)


if __name__ == '__main__':
    main()
