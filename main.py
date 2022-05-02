import itertools

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


def get_panel(latitude, longitude):
    if latitude is None or longitude is None:
        return None
    # when close to the horizontal plane, must be panels 1 through 4 (inclusive)
    if -np.pi / 4 <= latitude <= np.pi / 4:
        if -np.pi / 4 <= longitude <= np.pi / 4:
            return 1
        elif np.pi / 4 <= longitude <= 3 * np.pi / 4:
            return 2
        elif 3 * np.pi / 4 <= longitude or longitude <= -3 * np.pi / 4:
            return 3
        elif -3 * np.pi / 4 <= longitude <= -np.pi / 4:
            return 4
    # above a certain latitude, in panel 5
    elif latitude > np.pi / 4:
        return 5
    # below a certain latitude, in panel 6
    elif latitude < -np.pi / 4:
        return 6


def get_cube_coords(latitude, longitude):
    panel = get_panel(latitude, longitude)
    if panel is None:
        # TODO: come up with a more sensible way of handling these
        panel, x, y = np.nan, np.nan, np.nan
    else:
        if panel <= 4:
            offset = (((panel - 1) / 2) * np.pi)
            x = longitude - offset
            y = np.arctan(np.tan(latitude) / np.cos(longitude - offset))
        elif panel == 5:
            x = np.arctan(np.sin(longitude) / np.tan(latitude))
            y = np.arctan(-np.cos(longitude) / np.tan(latitude))
        elif panel == 6:
            x = np.arctan(-np.sin(longitude) / np.tan(latitude))
            y = np.arctan(-np.cos(longitude) / np.tan(latitude))
    return panel, x, y


def plot_sphere(measurement_positions):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Format data.
    x, y, z = [], [], []

    for proj_angle in measurement_positions.keys():
        vert_angles = measurement_positions[proj_angle].compressed()

        # convert from degrees to radians
        proj_angle = proj_angle * np.pi / 180
        vert_angles = vert_angles * np.pi / 180

        # convert to cartesian coordinates
        x_i = np.cos(vert_angles) * np.cos(proj_angle)
        y_i = np.cos(vert_angles) * np.sin(proj_angle)
        z_i = np.sin(vert_angles)

        x = x + x_i.tolist()
        y = y + y_i.tolist()
        z = z + z_i.tolist()

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    # Plot the surface.
    surf = ax.scatter(x, y, z, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()


class CubedSphere(object):
    def __init__(self, proj_angle, vert_angles):
        # convert degrees to radians by multiplying by a factor of pi/180
        self.proj_angle = proj_angle * np.pi / 180
        self.vert_angles = vert_angles * np.pi / 180

        self.cube_coords = torch.tensor(
            list(map(lambda x: get_cube_coords(latitude=x, longitude=self.proj_angle), self.vert_angles)))

        print(f"proj_angle: {self.proj_angle}")
        print(f"vert_angles shape: {self.vert_angles.shape}")
        print(f"cube_coords: {self.cube_coords}")
        print(f"cube_coords shape: {self.cube_coords.shape}")


def main():
    ds: ARI = load_data(data_folder='ARI', load_function=ARI, domain='magnitude_db', side='left')
    print(len(ds))

    plot_sphere(ds._selected_angles)
    # need to use protected member to get this data, no getters
    for angle in ds._selected_angles.keys():
        if angle == 0.0:
            CubedSphere(proj_angle=angle, vert_angles=ds._selected_angles[angle])


if __name__ == '__main__':
    main()
