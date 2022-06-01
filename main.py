import pickle
import torch
from hrtfdata.torch.full import ARI, CHEDAR
from pathlib import Path
import numpy as np

from cubed_sphere import CubedSphere
from plot import plot_3d_shape, plot_flat_cube, plot_impulse_response, plot_interpolated_features, plot_ir_subplots, \
    plot_original_features, plot_padded_panels
from convert_coordinates import convert_cube_to_sphere, convert_sphere_to_cube, convert_sphere_to_cartesian, \
    convert_cube_to_cartesian
from utils import get_feature_for_point, generate_euclidean_cube, triangle_encloses_point, get_possible_triangles, \
    calc_all_interpolated_features, save_euclidean_cube, calc_hrtf, pad_cubed_sphere, interpolate_fft_pad_all
from KalmanFilter import KalmanFilter as kf

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}})  # ,
    # subject_ids="last")


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
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='both')
    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    # generate_euclidean_cube(cs.get_sphere_coords(), "ARI_projection_4", edge_len=4)

    with open("ARI_projection_4", "rb") as file:
        load_cube, load_sphere, load_sphere_triangles, load_sphere_coeffs = pickle.load(file)

    edge_len = 4
    pad_width = 1
    all_subects = []
    for subject in range(len(ds)):
        print(f"Subject {subject} out of {len(ds)} ({round(100 * subject / len(ds))}%)")
        all_subects.append(interpolate_fft_pad_all(cs, ds[subject],
                                                   load_sphere, load_sphere_triangles, load_sphere_coeffs, load_cube,
                                                   edge_len, pad_width))

    for mag_tensor in all_subects:
        plot_padded_panels(torch.select(mag_tensor, 3, 5), edge_len, pad_width=pad_width,
                           label_cells=False,
                           title=f"All cube faces, with padded areas shown outside hashes ({subject})")


if __name__ == '__main__':
    main()
