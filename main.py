import cmath
import itertools
import pickle

import pandas as pd
import scipy.fft
import torch
from hrtfdata.torch.full import ARI, CHEDAR
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from matplotlib import cm, pyplot as plt
from collections import Counter

from barycentric_calcs import calc_all_distances, calc_barycentric_coordinates
from cubed_sphere import CubedSphere
from plot import plot_3d_shape, plot_flat_cube, plot_impulse_response, plot_interpolated_features, plot_ir_subplots, \
    plot_original_features
from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cube, convert_sphere_to_cartesian, \
    convert_cube_to_cartesian
from utils import get_feature_for_point, generate_euclidean_cube, triangle_encloses_point, get_possible_triangles, \
    calc_all_interpolated_features, save_euclidean_cube, calc_hrtf
from KalmanFilter import KalmanFilter as kf

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="last")  # temporary measure to avoid loading entire dataset each time


def remove_itd(hrir, pre_window, length):
    # normalize such that max(abs(hrir)) == 1
    rescaling_factor = 1 / max(np.abs(hrir))
    normalized_hrir = rescaling_factor * hrir

    # initialise Kalman filter
    x = np.array([[0]])  # estimated initial state
    p = np.array([[0]])  # estimated initial variance

    h = np.array([[1]])  # observation model (observation represents internal state directly)

    # r and q may require tuning
    r = np.array([[np.sqrt(400)]])  # variance of the observation noise
    q = np.array([[0.01]])  # variance of the process noise

    hrir_filter = kf(x, p, h, q, r)
    f = np.array([[1]])  # F is state transition model
    for i, z in enumerate(normalized_hrir):
        hrir_filter.prediction(f)
        hrir_filter.update(z)
        # find first time post fit residual exceeds some threshold
        if np.abs(hrir_filter.get_post_fit_residual()) > 0.005:
            over_threshold_index = i
            break

    # create fade window in order to taper off HRIR towards the beginning and end
    fadeout_len = 50
    fadeout_interval = -1. / fadeout_len
    fadeout = np.arange(1 + fadeout_interval, fadeout_interval, fadeout_interval).tolist()

    fadein_len = 10
    fadein_interval = 1. / fadein_len
    fadein = np.arange(0.0, 1.0, fadein_interval).tolist()

    # trim HRIR based on first time threshold is exceeded
    start = over_threshold_index - pre_window
    stop = start + length

    if len(hrir) >= stop:
        trimmed_hrir = hrir[start:stop]
        fade_window = fadein + [1] * (length - fadein_len - fadeout_len) + fadeout
        faded_hrir = trimmed_hrir * fade_window
    else:
        trimmed_hrir = hrir[start:]
        fade_window = fadein + [1] * (len(trimmed_hrir) - fadein_len - fadeout_len) + fadeout
        faded_hrir = trimmed_hrir * fade_window
        zero_pad = [0] * (length - len(trimmed_hrir))
        faded_hrir = np.ma.append(faded_hrir, zero_pad)

    return faded_hrir


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left')
    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    # 90 degree azimuth
    i = list(ds._selected_angles.keys()).index(90.0)
    # 0 degrees elevation
    j = list(ds._selected_angles[90.0]).index(0.0)
    if not ds[0]['features'].mask[i][j][0]:
        hrir = ds[0]['features'][i][j]
        transformed_hrir = remove_itd(hrir, pre_window=10, length=256)
        plot_ir_subplots(hrir, transformed_hrir, title1='Original', title2='With ITD removed', suptitle='HRIR')

        # hrtf = scipy.fft.fft(transformed_hrir)
    # generate_euclidean_cube(cs.get_sphere_coords(), "euclidean_data_ARI", edge_len=2)
    #
    # with open("euclidean_data_ARI", "rb") as file:
    #     load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)
    #
    # plot_interpolated_features(cs, ds[0]['features'], 30, load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs)


if __name__ == '__main__':
    main()
