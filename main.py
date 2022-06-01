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
    plot_original_features, plot_padded_panel, plot_padded_panels
from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cube, convert_sphere_to_cartesian, \
    convert_cube_to_cartesian
from utils import get_feature_for_point, generate_euclidean_cube, triangle_encloses_point, get_possible_triangles, \
    calc_all_interpolated_features, save_euclidean_cube, calc_hrtf, pad_cubed_sphere
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
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='right')
    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    # generate_euclidean_cube(cs.get_sphere_coords(), "ARI_projection_4", edge_len=4)

    with open("ARI_projection_4", "rb") as file:
        load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)

    # interpolated_hrirs is a list of interpolated HRIRs corresponding to the points specified in load_sphere and
    # load_cube, all three lists share the same ordering
    interpolated_hrirs = calc_all_interpolated_features(cs, ds[0]['features'],
                                                        load_sphere, load_sphere_triangles, load_sphere_coeffs)
    magnitudes, phases = calc_hrtf(interpolated_hrirs)

    edge_len = 4
    pad_width = 0
    tensor_width = edge_len + 2 * pad_width
    # create empty list of lists of lists
    magnitudes_list = [[[[] for _ in range(tensor_width)] for _ in range(tensor_width)] for _ in range(5)]

    count = 0
    for panel, x, y in load_cube:
        # based on cube coordinates, get indices for magnitudes list of lists
        i = panel - 1
        j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
        k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))

        # TODO: remove the 0 index from the magnitudes -- this is only for testing purposes
        magnitudes_list[i][j + pad_width][k + pad_width] = magnitudes[count]

        count += 1

    magnitudes_pad = pad_cubed_sphere(magnitudes_list)

    # convert list of numpy arrays into a single array, such that converting into tensor is faster
    magnitudes_tensor = torch.tensor(np.array(magnitudes_pad))

    print(magnitudes_tensor.shape)
    # print(torch.select(magnitudes_tensor, 3, 0).shape)

    # for panel in range(5):
    #     plot_padded_panel(magnitudes_tensor[panel], edge_len, f"Panel {panel} with padding")
    # plot_padded_panels(magnitudes_tensor, edge_len, "All cube faces, with padded areas shown outside hashes")

    plot_padded_panels(torch.select(magnitudes_tensor, 3, 100), edge_len, "All cube faces, with padded areas shown outside hashes")
    print([round(x[0], 4) for x in magnitudes[64:80]])
    print(load_cube[64:80])


if __name__ == '__main__':
    main()
