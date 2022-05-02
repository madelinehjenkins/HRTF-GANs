import itertools
import pandas as pd
import torch
from hrtfdata.torch.full import ARI
from hrtfdata.torch import collate_dict_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def load_data(data_folder, load_function, domain, side):
    base_dir = Path('/Users/madsjenkins/Imperial/HRTF/Volumes/home/HRTF Datasets')
    return load_function(base_dir / data_folder,
                         feature_spec={"hrirs": {'side': side, 'domain': domain}},
                         target_spec={"side": {}},
                         group_spec={"subject": {}},
                         subject_ids="first")  # temporary measure to avoid loading entire dataset each time


def get_panel(elevation, azimuth):
    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -np.pi / 4 <= elevation < np.pi / 4:
        if -np.pi / 4 <= azimuth < np.pi / 4:
            return 1
        elif np.pi / 4 <= azimuth < 3 * np.pi / 4:
            return 2
        elif 3 * np.pi / 4 <= azimuth < 5 * np.pi / 4:
            return 3
        elif 5 * np.pi / 4 <= azimuth:
            return 4
    # above a certain elevation, in panel 5
    elif elevation >= np.pi / 4:
        return 5
    # below a certain elevation, in panel 6
    elif elevation < -np.pi / 4:
        return 6


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
            offset = (((panel - 1) / 2) * np.pi)
            x = azimuth - offset
            y = np.arctan(np.tan(elevation) / np.cos(azimuth - offset))
        elif panel == 5:
            x = np.arctan(np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
        elif panel == 6:
            x = np.arctan(-np.sin(azimuth) / np.tan(elevation))
            y = np.arctan(-np.cos(azimuth) / np.tan(elevation))
    return panel, x, y


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
    ax.scatter(x, y, z, c=shading, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

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

            if panel == 4:
                x_i, y_i, z_i = p, -np.pi / 4, q
            elif panel == 1:
                x_i, y_i, z_i = np.pi / 4, p, q
            elif panel == 2:
                x_i, y_i, z_i = -p, np.pi / 4, q
            elif panel == 3:
                x_i, y_i, z_i = -np.pi / 4, -p, q
            elif panel == 5:
                x_i, y_i, z_i = -q, p, np.pi / 4
            elif panel == 6:
                x_i, y_i, z_i = q, p, -np.pi / 4

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask


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
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='magnitude_db', side='left')

    # need to use protected member to get this data, no getters
    cs = CubedSphere(sphere_coords=ds._selected_angles)

    make_3d_plot("sphere", cs.get_sphere_coords(), shading=cs.get_all_coords()["azimuth"])
    make_3d_plot("cube", cs.get_cube_coords(), shading=cs.get_all_coords()["azimuth"])


if __name__ == '__main__':
    main()
