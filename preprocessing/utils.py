import cmath
import pickle

import numpy as np
import scipy
import torch

from preprocessing.barycentric_calcs import calc_barycentric_coordinates, get_triangle_vertices
from preprocessing.convert_coordinates import convert_cube_to_sphere
from preprocessing.KalmanFilter import KalmanFilter
from preprocessing.padding import pad_cubed_sphere

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side, subject_ids=None):
    """Wrapper for the data loading functions from the hrtfdata package"""
    if subject_ids:
        return load_function(data_folder,
                             feature_spec={"hrirs": {'side': side, 'domain': domain}},
                             target_spec={"side": {}},
                             group_spec={"subject": {}}, subject_ids=subject_ids)
    return load_function(data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}})


def generate_euclidean_cube(measured_coords, filename, edge_len=16):
    """Calculate barycentric coordinates for projection based on a specified cube sphere edge length and a set of
    measured coordinates, finally save them to the file"""
    cube_coords, sphere_coords = [], []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                cube_coords.append((panel, x_i, y_i))
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))

    euclidean_sphere_triangles = []
    euclidean_sphere_coeffs = []

    for count, p in enumerate(sphere_coords):
        triangle_vertices = get_triangle_vertices(elevation=p[0], azimuth=p[1], sphere_coords=measured_coords)
        coeffs = calc_barycentric_coordinates(elevation=p[0], azimuth=p[1], closest_points=triangle_vertices)
        euclidean_sphere_triangles.append(triangle_vertices)
        euclidean_sphere_coeffs.append(coeffs)

        print(f"Data point {count} out of {len(sphere_coords)} ({round(100 * count / len(sphere_coords))}%)")

    # save euclidean_cube, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs
    with open(filename, "wb") as file:
        pickle.dump((cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs), file)


def save_euclidean_cube(edge_len=16):
    """Save euclidean cube as a txt file for use as input to matlab"""
    sphere_coords = []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))
    with open('../projection_coordinates/generated_coordinates.txt', 'w') as f:
        for coord in sphere_coords:
            print(coord)
            f.write(str(coord[0]))
            f.write(", ")
            f.write(str(coord[1]))
            f.write('\n')


def get_feature_for_point(elevation, azimuth, all_coords, subject_features):
    """For a given point (elevation, azimuth), get the associated feature value"""
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    azimuth_index = int(all_coords_row.azimuth_index)
    elevation_index = int(all_coords_row.elevation_index)
    return subject_features[azimuth_index][elevation_index]


def calc_interpolated_feature(triangle_vertices, coeffs, all_coords, subject_features):
    """Calculate the interpolated feature for a given point based on vertices specified by triangle_vertices, features
    specified by subject_features, and barycentric coefficients specified by coeffs"""
    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in triangle_vertices:
        features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
        features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]

    return interpolated_feature


def calc_all_interpolated_features(cs, features, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs):
    """Essentially a wrapper function for calc_interpolated_features above, calculated interpolated features for all
    points on the euclidean sphere rather than a single point"""
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            features_p = calc_interpolated_feature(triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=cs.get_all_coords(),
                                                   subject_features=features)

            selected_feature_interpolated.append(features_p)
        else:
            selected_feature_interpolated.append(None)

    return selected_feature_interpolated


def calc_hrtf(hrirs):
    """FFT to obtain HRTF from HRIR"""
    magnitudes = []
    phases = []
    for hrir in hrirs:
        hrtf = scipy.fft.fft(hrir)
        magnitude = abs(hrtf)
        phase = [cmath.phase(x) for x in hrtf]
        magnitudes.append(magnitude)
        phases.append(phase)

    return magnitudes, phases


def interpolate_fft_pad(cs, features, load_sphere, load_sphere_triangles, load_sphere_coeffs, load_cube, edge_len,
                        pad_width):
    """Combine all data processing steps into one function"""
    # interpolated_hrirs is a list of interpolated HRIRs corresponding to the points specified in load_sphere and
    # load_cube, all three lists share the same ordering
    interpolated_hrirs = calc_all_interpolated_features(cs, features,
                                                        load_sphere, load_sphere_triangles, load_sphere_coeffs)
    magnitudes, phases = calc_hrtf(interpolated_hrirs)

    # create empty list of lists of lists and initialize counter
    magnitudes_raw = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    count = 0

    for panel, x, y in load_cube:
        # based on cube coordinates, get indices for magnitudes list of lists
        i = panel - 1
        j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
        k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))

        # add to list of lists of lists and increment counter
        magnitudes_raw[i][j][k] = magnitudes[count]
        count += 1

    # pad each panel of the cubed sphere appropriately
    magnitudes_pad = pad_cubed_sphere(magnitudes_raw, pad_width)

    # convert list of numpy arrays into a single array, such that converting into tensor is faster
    return torch.tensor(np.array(magnitudes_pad))


def remove_itd(hrir, pre_window, length):
    """Remove ITD from HRIR using kalman filter"""
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

    hrir_filter = KalmanFilter(x, p, h, q, r)
    f = np.array([[1]])  # F is state transition model
    for i, z in enumerate(normalized_hrir):
        hrir_filter.prediction(f)
        hrir_filter.update(z)
        # find first time post fit residual exceeds some threshold
        if np.abs(hrir_filter.get_post_fit_residual()) > 0.005:
            over_threshold_index = i
            break
    else:
        raise RuntimeError("Kalman Filter did not find a time where post fit residual exceeded threshold.")

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
