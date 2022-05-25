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
    residuals = []
    for z in normalized_hrir:
        f = np.array([[1]])  # F is state transition model
        hrir_filter.prediction(f)
        hrir_filter.update(z)
        residuals.append(hrir_filter.get_post_fit_residual())

    # find first time post fit residual exceeds some threshold
    over_threshold_index = list(residuals).index(next(i for i in residuals if i > 0.005))

    # trim HRIR based on first time threshold is exceeded
    start = over_threshold_index - pre_window
    stop = start + length
    trimmed_hrir = hrir[start:stop]

    # create fade window in order to taper off HRIR towards the end
    fadeout = np.arange(0.9, -0.1, -0.1).tolist()
    fade_window = [1] * (length - 10) + fadeout
    return trimmed_hrir * fade_window


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    i = 104
    j = 5
    if not ds[0]['features'].mask[i][j][0]:
        hrir = ds[0]['features'][i][j]
        plot_impulse_response(hrir, title="Original HRIR")
        transformed_hrir = remove_itd(hrir, pre_window=3, length=30)
        plot_impulse_response(transformed_hrir, title="Trimmed and faded HRIR")
        hrtf = scipy.fft.fft(transformed_hrir)
        print(hrtf)
    # generate_euclidean_cube(cs.get_sphere_coords(), "euclidean_data_ARI", edge_len=2)
    #
    # with open("euclidean_data_ARI", "rb") as file:
    #     load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)
    #
    # plot_interpolated_features(cs, ds[0]['features'], 30, load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs)


if __name__ == '__main__':
    main()
