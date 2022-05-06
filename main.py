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
    ax.scatter(x, y, z, c=shading, s=10,
               linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

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
    ax.hlines(y=-3 * PI_4, xmin=-PI_4, xmax=PI_4, linewidth=2, color="grey")
    ax.hlines(y=3 * PI_4, xmin=-PI_4, xmax=PI_4, linewidth=2, color="grey")

    ax.vlines(x=-3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=-PI_4, ymin=-3 * PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=PI_4, ymin=-3 * PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
    ax.vlines(x=3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
    ax.vlines(x=5 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")

    # Plot the surface.
    ax.scatter(x, y, c=shading, s=10,
               linewidth=0, antialiased=False)

    fig.tight_layout()
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


def get_three_closest(elevation, azimuth, sphere_coords):
    distances = []
    for elev, azi in sphere_coords:
        if elev is not None and azi is not None:
            dist = calc_dist_haversine(elevation1=elevation, azimuth1=azimuth,
                                       elevation2=elev, azimuth2=azi)
            distances.append((elev, azi, dist))

    # list of (elevation, azimuth, distance) for the 3 closest points
    return sorted(distances, key=lambda x: x[2])[:3]


class CubedSphere(object):
    def __init__(self, sphere_coords):
        self.sphere_coords = []
        for proj_angle_i in sphere_coords.keys():
            # convert degrees to radians by multiplying by a factor of pi/180
            vert_angles_i = sphere_coords[proj_angle_i] * np.pi / 180
            proj_angle_i = proj_angle_i * np.pi / 180

            num_measurements = vert_angles_i.shape[0]
            # measurement_positions is stored as (elevation, azimuth)
            self.sphere_coords += list(zip(vert_angles_i.tolist(), [proj_angle_i] * num_measurements))

        # self.cube_coords is created from measurement_positions, such that order is the same
        self.cube_coords = list(itertools.starmap(get_cube_coords, self.sphere_coords))

        # create pandas dataframe containing all coordinate data (spherical and cubed sphere)
        # this can be useful for debugging
        self.all_coords = pd.concat([pd.DataFrame(self.sphere_coords, columns=["elevation", "azimuth"]),
                                     pd.DataFrame(self.cube_coords, columns=["panel", "x", "y"])], axis="columns")

    def get_sphere_coords(self):
        return self.sphere_coords

    def get_cube_coords(self):
        return self.cube_coords

    def get_all_coords(self):
        return self.all_coords


def main():
    # ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='magnitude_db', side='left')
    ds: CHEDAR = load_data(data_folder='CHEDAR', load_function=CHEDAR, domain='magnitude_db', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)
    euclidean_cube, euclidean_sphere = generate_euclidean_cube()

    print(euclidean_sphere[0])
    print(get_three_closest(elevation=euclidean_sphere[0][0], azimuth=euclidean_sphere[0][1], sphere_coords=cs.get_sphere_coords()))

    # shading_feature = "azimuth"
    # make_3d_plot("sphere", cs.get_sphere_coords(), shading=cs.get_all_coords()[shading_feature])
    # make_3d_plot("cube", cs.get_cube_coords(), shading=cs.get_all_coords()[shading_feature])
    # make_flat_cube_plot(cs.get_cube_coords(), shading=cs.get_all_coords()[shading_feature])
    #
    # make_flat_cube_plot(euclidean_cube)
    # make_3d_plot("cube", euclidean_cube)
    # make_3d_plot("sphere", euclidean_sphere)

    # all_coords = cs.get_all_coords()
    # print(f"all coords shape: {all_coords.shape}")
    # print(f"features shape: {ds[0]['features'].shape}")
    # print(f"subject 0, proj angle 0, vert angle 0, all HRTF frequencies: {ds[0]['features'][0][0]}")


if __name__ == '__main__':
    main()
