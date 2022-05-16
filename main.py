import itertools
import pandas as pd
import torch
from hrtfdata.torch.full import ARI, CHEDAR
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from collections import Counter

PI_4 = np.pi / 4


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="first")  # temporary measure to avoid loading entire dataset each time


def get_offset(quadrant):
    return ((quadrant - 1) / 2) * np.pi


def get_panel(elevation, azimuth):
    # use the azimuth to determine the quadrant of the sphere
    if azimuth < np.pi / 4:
        quadrant = 1
    elif azimuth < 3 * np.pi / 4:
        quadrant = 2
    elif azimuth < 5 * np.pi / 4:
        quadrant = 3
    else:
        quadrant = 4

    offset = get_offset(quadrant)
    threshold_val = np.tan(elevation) / np.cos(azimuth - offset)
    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -1 <= threshold_val < 1:
        return quadrant
    # above a certain elevation, in panel 5
    elif threshold_val >= 1:
        return 5
    # below a certain elevation, in panel 6
    elif threshold_val < -1:
        return 6


# used for obtaining cubed sphere coordinates from a pair of spherical coordinates
def get_cube_coords(elevation, azimuth):
    if elevation is None or azimuth is None:
        # if this position was not measured in the sphere, keep as np.nan in cube
        panel, x, y = np.nan, np.nan, np.nan
    else:
        # shift the range of azimuth angles such that it works with conversion equations
        if azimuth < -np.pi / 4:
            azimuth += 2 * np.pi
        panel = get_panel(elevation, azimuth)

        if panel <= 4:
            offset = get_offset(panel)
            x = azimuth - offset
            y = np.arctan(np.tan(elevation) / np.cos(azimuth - offset))
        elif panel == 5:
            x = np.arctan(np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
        elif panel == 6:
            x = np.arctan(-np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
    return panel, x, y


# used for obtaining spherical coordinates from cubed sphere coordinates
def get_sphere_coords(panel, x, y):
    if panel <= 4:
        offset = get_offset(panel)
        azimuth = x + offset
        elevation = np.arctan(np.tan(y) * np.cos(x))
    elif panel == 5:
        azimuth = np.arctan(-np.tan(x) / np.tan(y))
        elevation = np.arctan(np.sin(azimuth) / np.tan(x))
        if elevation < 0:
            elevation *= -1
            azimuth += np.pi
    # not including panel 6 for now, as it is being excluded from this data
    elif panel == 6:
        pass

    # ensure azimuth is in range -pi to +pi
    while azimuth > np.pi:
        azimuth -= 2 * np.pi
    while azimuth <= -np.pi:
        azimuth += 2 * np.pi

    return elevation, azimuth


def make_3d_plot(shape, coordinates, shading=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Format data.
    if shape == "sphere":
        x, y, z, mask = convert_sphere_to_cartesian(coordinates)
    elif shape == "cube":
        x, y, z, mask = convert_cube_to_cartesian(coordinates)

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # Plot the surface.
    sc = ax.scatter(x, y, z, c=shading, s=10,
                    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    plt.colorbar(sc)

    plt.show()


def make_flat_cube_plot(cube_coords, shading=None):
    fig, ax = plt.subplots()

    # Format data.
    x, y = [], []
    mask = []

    for panel, p, q in cube_coords:
        if not np.isnan(p) and not np.isnan(q):
            mask.append(True)

            if panel == 1:
                x_i, y_i = p, q
            elif panel == 2:
                x_i, y_i = p + np.pi / 2, q
            elif panel == 3:
                x_i, y_i = p + np.pi, q
            elif panel == 4:
                x_i, y_i = p - np.pi / 2, q
            elif panel == 5:
                x_i, y_i = p, q + np.pi / 2
            else:
                x_i, y_i = p, q - np.pi / 2

            x.append(x_i)
            y.append(y_i)
        else:
            mask.append(False)

    x, y = np.asarray(x), np.asarray(y)

    if shading is not None:
        shading = list(itertools.compress(shading, mask))

    # draw lines outlining cube
    ax.hlines(y=-PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
    ax.hlines(y=PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
    ax.hlines(y=3 * PI_4, xmin=-PI_4, xmax=PI_4, linewidth=2, color="grey")

    ax.vlines(x=-3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=-PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=5 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")

    # Plot the surface.
    sc = ax.scatter(x, y, c=shading, s=10,
                    linewidth=0, antialiased=False)
    plt.colorbar(sc)

    fig.tight_layout()
    fig.set_size_inches(9, 4)
    plt.show()


def convert_sphere_to_cartesian(coordinates):
    x, y, z = [], [], []
    mask = []

    for elevation, azimuth in coordinates:
        if elevation is not None and azimuth is not None:
            mask.append(True)
            # convert to cartesian coordinates
            x_i = np.cos(elevation) * np.cos(azimuth)
            y_i = np.cos(elevation) * np.sin(azimuth)
            z_i = np.sin(elevation)

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask


def convert_cube_to_cartesian(coordinates):
    x, y, z = [], [], []
    mask = []

    for panel, p, q in coordinates:
        if not np.isnan(p) and not np.isnan(q):
            mask.append(True)
            if panel == 1:
                x_i, y_i, z_i = PI_4, p, q
            elif panel == 2:
                x_i, y_i, z_i = -p, PI_4, q
            elif panel == 3:
                x_i, y_i, z_i = -PI_4, -p, q
            elif panel == 4:
                x_i, y_i, z_i = p, -PI_4, q
            elif panel == 5:
                x_i, y_i, z_i = -q, p, PI_4
            else:
                x_i, y_i, z_i = q, p, -PI_4

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask


def generate_euclidean_cube(edge_len=24):
    cube_coords, sphere_coords = [], []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                cube_coords.append((panel, x_i, y_i))
                sphere_coords.append(get_sphere_coords(panel, x_i, y_i))
    return cube_coords, sphere_coords


def calc_dist_haversine(elevation1, azimuth1, elevation2, azimuth2):
    # adapted from CalculateDistance_HaversineFomrula in the 3DTune-In toolkit
    # https://github.com/3DTune-In/3dti_AudioToolkit/blob/master/3dti_Toolkit/BinauralSpatializer/HRTF.cpp#L1052
    increment_azimuth = azimuth1 - azimuth2
    increment_elevation = elevation1 - elevation2
    sin2_inc_elev = np.sin(increment_elevation / 2) ** 2
    cos_elev1 = np.cos(elevation1)
    cos_elev2 = np.cos(elevation2)
    sin2_inc_azi = np.sin(increment_azimuth / 2) ** 2
    raiz = sin2_inc_elev + (cos_elev1 * cos_elev2 * sin2_inc_azi)
    sqrt_distance = raiz ** 0.5
    distance = np.arcsin(sqrt_distance)
    return distance


# get three closest measured point on sphere for barycentric interpolation
def get_three_closest(elevation, azimuth, sphere_coords):
    distances = []
    for elev, azi in sphere_coords:
        if elev is not None and azi is not None:
            dist = calc_dist_haversine(elevation1=elevation, azimuth1=azimuth,
                                       elevation2=elev, azimuth2=azi)
            distances.append((elev, azi, dist))

    # list of (elevation, azimuth, distance) for the 3 closest points
    return sorted(distances, key=lambda x: x[2])[:3]


# get alpha, beta, and gamma coeffs for barycentric interpolation
def calculate_alpha_beta_gamma(elevation, azimuth, closest_points):
    # not zero indexing var names in order to match equations in 3D Tune-In Toolkit paper
    elev1, elev2, elev3 = closest_points[0][0], closest_points[1][0], closest_points[2][0]
    azi1, azi2, azi3 = closest_points[0][1], closest_points[1][1], closest_points[2][1]

    # based on equation 5 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    denominator = (elev2 - elev3) * (azi1 - azi3) + (azi3 - azi2) * (elev1 - elev3)
    # TODO: how to handle denominator of zero?
    if denominator == 0:
        alpha, beta, gamma = 1. / 3, 1. / 3, 1. / 3
    else:
        alpha = ((elev2 - elev3) * (azimuth - azi3) + (azi3 - azi2) * (elevation - elev3)) / denominator
        beta = ((elev3 - elev1) * (azimuth - azi3) + (azi1 - azi3) * (elevation - elev3)) / denominator
        gamma = 1 - alpha - beta

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def get_feature_for_point(elevation, azimuth, all_coords, subject_features):
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    azimuth_index = int(all_coords_row.azimuth_index)
    elevation_index = int(all_coords_row.elevation_index)
    return subject_features[azimuth_index][elevation_index]


def calc_interpolated_feature(elevation, azimuth, sphere_coords, all_coords, subject_features):
    three_closest = get_three_closest(elevation=elevation, azimuth=azimuth, sphere_coords=sphere_coords)
    check = check_point_in_triangle(elevation, azimuth, [three_closest[0][:2], three_closest[1][:2], three_closest[2][:2]])
    # TODO: if check is false, look at next triangles
    coeffs = calculate_alpha_beta_gamma(elevation=elevation, azimuth=azimuth, closest_points=three_closest)

    print(f'is point in triangle? {check}')
    if not check and coeffs["alpha"] != 1./3:
        print(f'three closest: {three_closest}')
        print(f'point: {elevation,azimuth}')
        print(f'coeffs: {coeffs}\n')

    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in three_closest:
        features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
        features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]

    return interpolated_feature


def make_interpolated_plots(cs, features, feature_index):
    euclidean_cube, euclidean_sphere = generate_euclidean_cube()
    all_coords = cs.get_all_coords()

    selected_feature_raw = []
    for p in cs.get_sphere_coords():
        if p[0] is not None:
            features_p = get_feature_for_point(p[0], p[1], all_coords, features)
            selected_feature_raw.append(features_p[feature_index])
        else:
            selected_feature_raw.append(None)

    make_3d_plot("sphere", cs.get_sphere_coords(), shading=selected_feature_raw)
    make_3d_plot("cube", cs.get_cube_coords(), shading=selected_feature_raw)
    make_flat_cube_plot(cs.get_cube_coords(), shading=selected_feature_raw)

    selected_feature_interpolated = []
    for p in euclidean_sphere:
        if p[0] is not None:
            features_p = calc_interpolated_feature(elevation=p[0], azimuth=p[1],
                                                   sphere_coords=cs.get_sphere_coords(), all_coords=all_coords,
                                                   subject_features=features)
            selected_feature_interpolated.append(features_p[feature_index])
        else:
            selected_feature_interpolated.append(None)

    make_flat_cube_plot(euclidean_cube, shading=selected_feature_interpolated)
    make_3d_plot("cube", euclidean_cube, shading=selected_feature_interpolated)
    make_3d_plot("sphere", euclidean_sphere, shading=selected_feature_interpolated)


def plot_impulse_response(times):
    plt.plot(times)
    plt.show()


class CubedSphere(object):
    def __init__(self, sphere_coords):
        # initiate two lists of tuples, one will store (elevation, azimuth) for every measurement point
        # the other will store (elevation_index, azimuth_index) for every measurement point
        self.sphere_coords = []
        self.indices = []

        # at this stage, we can simplify by acting as if there are the same number of elevation measurement points at
        # every azimuth angle
        num_elevation_measurements = sphere_coords[0].shape[0]
        elevation_indices = list(range(num_elevation_measurements))

        # loop through all azimuth positions
        for azimuth_index, azimuth in enumerate(sphere_coords.keys()):
            # convert degrees to radians by multiplying by a factor of pi/180
            elevation = sphere_coords[azimuth] * np.pi / 180
            azimuth = azimuth * np.pi / 180

            # sphere_coords is stored as (elevation, azimuth). Ultimately, we're creating a list of (elevation,
            # azimuth) pairs for every measurement position in the sphere
            self.sphere_coords += list(zip(elevation.tolist(), [azimuth] * num_elevation_measurements))
            self.indices += list(zip(elevation_indices, [azimuth_index] * num_elevation_measurements))

        # self.cube_coords is created from measurement_positions, such that order is the same
        self.cube_coords = list(itertools.starmap(get_cube_coords, self.sphere_coords))

        # create pandas dataframe containing all coordinate data (spherical and cubed sphere)
        # this can be useful for debugging
        self.all_coords = pd.concat([pd.DataFrame(self.indices, columns=["elevation_index", "azimuth_index"]),
                                     pd.DataFrame(self.sphere_coords, columns=["elevation", "azimuth"]),
                                     pd.DataFrame(self.cube_coords, columns=["panel", "x", "y"])], axis="columns")

    def get_sphere_coords(self):
        return self.sphere_coords

    def get_cube_coords(self):
        return self.cube_coords

    def get_all_coords(self):
        return self.all_coords


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='time', side='left')
    # ds: CHEDAR = load_data(data_folder='CHEDAR', load_function=CHEDAR, domain='magnitude_db', side='left')

    # need to use protected member to get this data, no getters
    # cs = CubedSphere(sphere_coords=ds._selected_angles)
    # make_interpolated_plots(cs, ds[0]['features'], feature_index=20)

    # print(f'Features shape: {ds[0]["features"].shape}')
    # print(f'# of row angles, # of col angles: {len(ds.row_angles), len(ds.column_angles)}')
    # print(f'HRIR sample rate: {ds.hrir_samplerate}')
    # print(f'# of HRTF frequencies: {len(ds.hrtf_frequencies)}')
    # print(ds[0]["features"][0][0])

    # values = ds[0]["features"][0][20]
    for row in ds[0]["features"]:
        for col in row:
            values = max(range(len(col)), key=col.__getitem__)
            if values > 75:
                print(values)
                plot_impulse_response(col)
            if values == 37:
                print(values)
                plot_impulse_response(col)
    # plot_impulse_response(ds[0]["features"][70][10])

    # print(triangle_area(0, 0, 2, 0, 1, 6, 1, 1))
    # print(triangle_area(2, 0, -2, 2, 4, 2, 1, 1))
    # print(triangle_area(2, 0, -2, 2, 1, 6, 1, 1))


if __name__ == '__main__':
    main()
